# SPDX-License-Identifier: Apache-2.0
"""CPU behavior checks; run directly, without importing the NPU package."""

import ast
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(os.environ.get("ASCEND_SOURCE_ROOT", Path(__file__).resolve().parents[3]))


def methods(namespace):
    path = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMoERunner")
    selected = [
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_fused_mc2_schedule_supported", "_fused_mc2_scheduled_forward", "shared_forward_impl"}
    ]
    for n in selected:
        n.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.edges = {}
        self.ops = {}
        test = self

        class Stream:
            def __init__(self):
                self.tail = None

            def emit(self, name):
                node = len(test.edges)
                test.edges[node] = {self.tail} if self.tail is not None else set()
                self.tail = node
                test.ops.setdefault(name, []).append(node)
                return node

            def record_event(self):
                return self.emit("event")

            def wait_event(self, event):
                assert event in test.edges  # Every wait must follow a record.
                test.edges[self.emit("wait")].add(event)

        self.main, self.side = Stream(), Stream()
        self.current = self.main

        class Tensor:
            dtype = "bf16"

            def record_stream(self, stream):
                test.records.append(stream)

            def float(self):
                test.current.emit("cast")
                return self

        self.records = []
        self.Tensor = Tensor
        self.x = Tensor()

        @contextmanager
        def switch(stream):
            previous, self.current = self.current, stream
            try:
                yield
            finally:
                self.current = previous

        def quant(x):
            self.assertIs(x, self.x)
            self.current.emit("quant")
            return Tensor(), Tensor()

        def matmul(x, weight, weight_scale, **kw):
            self.current.emit(weight.name)
            self.assertEqual(kw["output_dtype"], "int32" if weight.name == "gateup" else "bf16")
            return Tensor()

        def activation(**kw):
            self.activation_args = kw
            self.current.emit("swiglu")
            return Tensor(), Tensor()

        def linear(x, weight):
            self.current.emit("gating")
            return Tensor()

        def routed(x, logits):
            self.assertIs(x, self.x)
            self.current.emit("topk")
            self.current.emit("dispatch")
            return "routed"

        def finalize(output):
            self.assertIs(self.current, self.main)
            self.current.emit("finalize")
            return "shared"

        self.ctx = NS(moe_comm_type="fused", use_mega_moe=False, flash_comm_v1_enabled=False)
        self.config = NS(enable_fused_mc2=1, mix_placement=False)
        self.ns = methods(
            {
                "torch": NS(
                    Tensor=Tensor,
                    int32="int32",
                    int8="int8",
                    npu=NS(current_stream=lambda: self.current),
                    ops=NS(_C_ascend=NS(npu_dequant_swiglu_quant=activation)),
                ),
                "torch_npu": NS(npu_dynamic_quant=quant, npu_quant_matmul=matmul),
                "F": NS(linear=linear),
                "shared_experts_calculation_stream": lambda: self.side,
                "npu_stream_switch": switch,
                "QuantType": NS(W8A8="w8", W4A8="w4"),
                "_EXTRA_CTX": self.ctx,
                "get_ascend_config": lambda: self.config,
                "MoECommType": NS(FUSED_MC2="fused"),
                "SituActivationConfig": type("Situ", (), {}),
                "AscendSituAndMul": type("SituFn", (), {}),
                "logger": NS(info_once=lambda *a: None, warning_once=lambda *a: None),
            }
        )
        self.runner = NS(
            quant_type="w8",
            gate=NS(weight_fp32="gate"),
            is_internal_router=True,
            fused_mc2_shared_schedule=True,
            activation="silu",
            _prepare_shared_expert_input=lambda x: x,
            _quant_method=NS(quant_method=type("AscendW8A8DynamicFusedMoEMethod", (), {})()),
            routed_experts=NS(swiglu_limit=7, swiglu_alpha=1.2, swiglu_beta=0.3),
            _shared_experts=NS(
                act_fn="silu",
                gate_up_proj=NS(weight=NS(name="gateup", dtype="int8"), weight_scale=1, weight_scale_fp32=2),
                down_proj=NS(weight=NS(name="down", dtype="int8"), weight_scale=3),
            ),
            no_shared_forward_impl=routed,
            _finalize_shared_expert_output=finalize,
        )

    def precedes(self, before, after):
        pending = list(self.edges[self.ops[after][0]])
        visited = set()
        while pending:
            node = pending.pop()
            if node == self.ops[before][0]:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(self.edges[node])
        return False

    def test_dag_and_math_parameters(self):
        result = self.ns["_fused_mc2_scheduled_forward"](self.runner, self.x, self.x)
        self.assertEqual(result, ("shared", "routed"))
        for a, b in [
            ("quant", "gateup"),
            ("gateup", "gating"),
            ("gateup", "swiglu"),
            ("gating", "down"),
            ("swiglu", "down"),
            ("gating", "topk"),
            ("topk", "dispatch"),
            ("down", "finalize"),
            ("dispatch", "finalize"),
        ]:
            self.assertTrue(self.precedes(a, b), (a, b))
        self.assertFalse(self.precedes("down", "dispatch"))
        self.assertFalse(self.precedes("dispatch", "down"))
        self.assertFalse(self.precedes("swiglu", "gating"))
        self.assertEqual(self.activation_args["clamp_limit"], 7)
        self.assertEqual(self.activation_args["glu_alpha"], 1.2)
        self.assertEqual(self.activation_args["glu_bias"], 0.3)
        self.assertEqual(self.records, [self.side, self.main])

    def test_w4_metadata_matches_builder(self):
        self.runner.quant_type = "w4"
        self.ns["_fused_mc2_scheduled_forward"](self.runner, self.x, self.x)
        self.assertEqual(self.activation_args["glu_alpha"], 1)
        self.assertEqual(self.activation_args["glu_bias"], 0)

    def test_support_and_fallback(self):
        supported = self.ns["_fused_mc2_schedule_supported"]
        self.assertTrue(supported(self.runner))
        for obj, attr, value in [
            (self.ctx, "moe_comm_type", "allgather"),
            (self.ctx, "use_mega_moe", True),
            (self.config, "enable_fused_mc2", 0),
            (self.config, "mix_placement", True),
            (self.runner, "is_internal_router", False),
            (self.runner, "quant_type", "fp8"),
            (self.runner, "activation", "gelu"),
            (self.runner._quant_method, "quant_method", object()),
            (self.runner._shared_experts.gate_up_proj.weight, "dtype", "int32"),
            (self.runner, "_shared_experts", None),
            (self.runner._shared_experts, "expert_gate", object()),
            (self.runner._shared_experts.gate_up_proj, "bias", object()),
            (self.runner.routed_experts, "_ascend_moe_lora_context", object()),
        ]:
            with self.subTest(attr=attr):
                old = getattr(obj, attr, None)
                setattr(obj, attr, value)
                self.assertFalse(supported(self.runner))
                setattr(obj, attr, old)

    def test_none_metadata_matches_builder(self):
        self.runner.routed_experts.swiglu_limit = None
        self.runner.routed_experts.swiglu_alpha = None
        self.runner.routed_experts.swiglu_beta = None
        self.ns["_fused_mc2_scheduled_forward"](self.runner, self.x, self.x)
        self.assertEqual(self.activation_args["clamp_limit"], 0)
        self.assertEqual(self.activation_args["glu_alpha"], 1)
        self.assertEqual(self.activation_args["glu_bias"], 0)

    def test_flashcomm_preparation_precedes_fork(self):
        self.ctx.flash_comm_v1_enabled = True
        self.assertTrue(self.ns["_fused_mc2_schedule_supported"](self.runner))

        def prepare(x):
            self.assertIs(self.current, self.main)
            self.main.emit("prepare")
            return x

        self.runner._prepare_shared_expert_input = prepare
        self.ns["_fused_mc2_scheduled_forward"](self.runner, self.x, self.x)
        self.assertTrue(self.precedes("prepare", "quant"))
        self.assertTrue(self.precedes("prepare", "gating"))

    def test_runner_entry_routes_to_schedule(self):
        self.runner._fused_mc2_schedule_supported = lambda: True
        self.runner._fused_mc2_scheduled_forward = lambda x, y: (x, y)
        other = self.Tensor()
        self.assertEqual(self.ns["shared_forward_impl"](self.runner, self.x, None, other), (self.x, other))


if __name__ == "__main__":
    unittest.main()
