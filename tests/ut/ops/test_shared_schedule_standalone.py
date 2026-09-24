# SPDX-License-Identifier: Apache-2.0
"""Run directly without repository conftest or NPU dependencies."""

import __future__

import ast
import importlib.util
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

ROOT = Path(os.environ.get("ASCEND_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
spec = importlib.util.spec_from_file_location("schedule", ROOT / "vllm_ascend/ops/fused_moe/shared_schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class Stream:
    def __init__(self, name):
        self.name = name
        self.clock = 0
        self.dependencies = set()

    def record_event(self):
        return self.dependencies.copy()

    def wait_event(self, event):
        assert isinstance(event, set), "event must have been recorded"
        self.dependencies.update(event)

    def wait_stream(self, stream):
        self.dependencies.update(stream.dependencies)


class Tensor:
    dtype = "bf16"

    def record_stream(self, stream):
        self.owner = stream


class Backend:
    def __init__(self):
        self.main = Stream("main")
        self.shared = Stream("shared")
        self.active = self.main
        self.before = {}

    def current_stream(self):
        return self.active

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous

    def op(self, name):
        self.before[name] = self.active.dependencies.copy()
        self.active.dependencies.add(name)


class TestSchedule(unittest.TestCase):
    def test_eligibility_and_fallback(self):
        path = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_can_schedule_fused_shared"
        )
        context = NS(moe_comm_type="fused", use_mega_moe=False)
        log = MagicMock()
        namespace = {
            "torch": NS(int8="int8"),
            "_EXTRA_CTX": context,
            "MoECommType": NS(FUSED_MC2="fused"),
            "QuantType": NS(W8A8="w8", W4A8="w4"),
            "AscendSituAndMul": type("Situ", (), {}),
            "get_ascend_config": lambda: NS(enable_fused_mc2=1),
            "logger": log,
        }
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        projection = NS(weight=NS(dtype="int8"), weight_scale=1, weight_scale_fp32=1)
        owner = NS(
            fused_mc2_shared_schedule=True,
            is_internal_router=True,
            quant_type="w8",
            activation="silu",
            routed_experts=NS(),
            _shared_experts=NS(act_fn=object(), gate_up_proj=projection, down_proj=projection),
            _quant_method=NS(quant_method=type("AscendW8A8DynamicFusedMoEMethod", (), {})()),
        )
        eligible = namespace[method.name]
        self.assertTrue(eligible(owner))
        for field, value in (("use_mega_moe", True), ("moe_comm_type", "allgather")):
            old = getattr(context, field)
            setattr(context, field, value)
            self.assertFalse(eligible(owner))
            setattr(context, field, old)
        owner.is_internal_router = False
        self.assertFalse(eligible(owner))
        self.assertEqual(log.warning_once.call_count, 3)
        owner.fused_mc2_shared_schedule = False
        self.assertFalse(eligible(owner))
        self.assertEqual(log.warning_once.call_count, 3)

    def test_integer_backends_call_hook_after_topk_before_fusion(self):
        for name, cls_name in (
            ("w8a8_dynamic", "AscendW8A8DynamicFusedMoEMethod"),
            ("w4a8", "AscendW4A8DynamicFusedMoEMethod"),
        ):
            with self.subTest(backend=name):
                path = ROOT / f"vllm_ascend/quantization/methods/{name}.py"
                tree = ast.parse(path.read_text(encoding="utf-8"))
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
                method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "apply")
                order = []
                tensor = MagicMock()
                tensor.shape = (6, 256)
                tensor.to.return_value = tensor

                def select(order=order, tensor=tensor, **kwargs):
                    order.append("topk")
                    return tensor, tensor

                def fused(order=order, **kwargs):
                    order.append("fusion")
                    return "out"

                namespace = {
                    "torch": NS(tensor=lambda *a, tensor=tensor, **k: tensor, float32="fp32"),
                    "get_moe_num_logical_experts": lambda *a, **k: 256,
                    "select_experts": select,
                    "_EXTRA_CTX": NS(
                        moe_comm_type="fused", use_mega_moe=False, moe_comm_method=NS(fused_experts=fused)
                    ),
                    "MoECommType": NS(FUSED_MC2="fused"),
                    "get_ascend_config": lambda: NS(enable_fused_mc2=1),
                    "build_fused_experts_input": lambda **kwargs: kwargs,
                }
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]),
                        str(path),
                        "exec",
                        flags=__future__.annotations.compiler_flag,
                    ),
                    namespace,
                )
                layer = NS(
                    w13_weight=tensor,
                    w2_weight=tensor,
                    w13_weight_scale=tensor,
                    w2_weight_scale=tensor,
                    fused_w1_scale=tensor,
                    fused_w2_scale=tensor,
                    swiglu_limit=0.0,
                )
                owner = NS(dynamic_eplb=False, in_dtype="bf16", quant_type=name, is_per_channel_weight=True)
                result = namespace["apply"](
                    owner, layer, tensor, tensor, 8, True, before_fused_experts=lambda order=order: order.append("hook")
                )
                self.assertEqual(result, "out")
                self.assertEqual(order, ["topk", "hook", "fusion"])

    def test_runner_and_adapter_callback_forwarding(self):
        def load(relative, class_name, method_name, namespace):
            path = ROOT / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
            exec(
                compile(
                    ast.Module(body=[method], type_ignores=[]),
                    str(path),
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                namespace,
            )
            return namespace[method_name]

        adapter = load("vllm_ascend/quantization/method_adapters.py", "AscendFusedMoEMethod", "apply", {})
        backend = MagicMock()
        backend.apply.return_value = NS(routed_out="raw")
        adapter_owner = NS(quant_method=backend, tid2eid=None)
        owner = MagicMock()
        owner.enable_npugraph_ex_static_kernel = False
        owner.dynamic_eplb = False
        owner.routed_experts._ascend_moe_lora_context = None
        owner._quant_method.apply.side_effect = lambda **kwargs: adapter(adapter_owner, **kwargs)
        comm = MagicMock()
        comm.prepare.return_value = NS(
            hidden_states="prepared",
            router_logits="logits",
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )
        comm.finalize.return_value = "finalized"
        namespace = {
            "get_forward_context": lambda: NS(),
            "_EXTRA_CTX": NS(in_profile_run=False, flash_comm_v1_enabled=False, moe_comm_method=comm),
            "AllGatherCommImpl": type("AllGather", (), {}),
        }
        runner = load("vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", "no_shared_forward_impl", namespace)
        callback = lambda: None
        self.assertEqual(runner(owner, "input", "router", before_fused_experts=callback), "finalized")
        self.assertIs(backend.apply.call_args.kwargs["before_fused_experts"], callback)
        self.assertEqual(backend.apply.call_args.kwargs["x"], "prepared")
        runner(owner, "input", "router")
        self.assertNotIn("before_fused_experts", backend.apply.call_args.kwargs)

    def test_scheme3_dependencies_and_output_join(self):
        backend = Backend()
        inputs, output = Tensor(), Tensor()

        def stages():
            for name in ("quant", "gateup", "swiglu", "down"):
                backend.op(name)
                yield output

        def gate():
            backend.op("gating")
            return "logits"

        def route(logits, callback):
            self.assertEqual(logits, "logits")
            backend.op("topk")
            callback()
            backend.op("dispatch")
            backend.op("finalize")
            return "routed"

        self.assertEqual(
            schedule.run_scheme3(backend, backend.shared, inputs, stages(), gate, route), (output, "routed")
        )
        self.assertNotIn("gating", backend.before["quant"])
        self.assertIn("gating", backend.before["gateup"])
        self.assertNotIn("topk", backend.before["gateup"])
        self.assertIn("topk", backend.before["swiglu"])
        self.assertNotIn("dispatch", backend.before["swiglu"])
        self.assertIn("finalize", backend.before["down"])
        self.assertIn("down", backend.main.dependencies)
        self.assertIs(inputs.owner, backend.shared)
        self.assertIs(output.owner, backend.main)

    def test_missing_callback_fails(self):
        backend = Backend()
        with self.assertRaisesRegex(RuntimeError, "not called"):
            schedule.run_scheme3(backend, backend.shared, Tensor(), iter([None] * 4), lambda: None, lambda *_: None)

    def test_actual_shared_math_parameters(self):
        path = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_scheduled_shared_steps"
        )
        calls = []

        def matmul(*args, **kwargs):
            calls.append((args, kwargs))
            return Tensor()

        def swiglu(**kwargs):
            calls.append(kwargs)
            return Tensor(), "output_scale"

        namespace = {
            "torch": NS(int32="int32", ops=NS(_C_ascend=NS(npu_dequant_swiglu_quant=swiglu))),
            "torch_npu": NS(npu_dynamic_quant=lambda x: (Tensor(), "input_scale"), npu_quant_matmul=matmul),
            "QuantType": NS(W8A8="w8", W4A8="w4"),
        }
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        shared = NS(
            gate_up_proj=NS(weight="up", weight_scale="up_scale", weight_scale_fp32="fp32"),
            down_proj=NS(weight="down", weight_scale="down_scale"),
        )
        owner = NS(
            _shared_experts=shared,
            quant_type="w8",
            routed_experts=NS(swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None),
        )
        result = list(namespace["_scheduled_shared_steps"](owner, Tensor()))
        self.assertEqual(len(result), 4)
        self.assertEqual(calls[0][1]["output_dtype"], "int32")
        self.assertEqual(calls[1]["activation_scale"], "input_scale")
        self.assertEqual((calls[1]["clamp_limit"], calls[1]["glu_alpha"], calls[1]["glu_bias"]), (0.0, 1.0, 0.0))
        self.assertEqual(calls[2][1]["pertoken_scale"], "output_scale")
        self.assertEqual(calls[2][1]["output_dtype"], "bf16")


if __name__ == "__main__":
    unittest.main()
