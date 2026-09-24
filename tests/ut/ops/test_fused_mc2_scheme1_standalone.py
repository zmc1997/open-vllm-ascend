"""CPU checks of the actual runner methods; no torch or NPU imports required."""

import argparse
import ast
import math
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS


def load_method(root, path, cls, name, namespace):
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(root / path), "exec"), namespace)
    return namespace[name]


class Tensor:
    dtype = "bf16"

    def __init__(self, value, deps=()):
        self.value = value
        self.deps = set(deps)
        self.streams = []

    def float(self):
        return self

    def record_stream(self, stream):
        self.streams.append(stream)


class Stream:
    def __init__(self):
        self.deps = set()

    def record_event(self):
        return frozenset(self.deps)

    def wait_event(self, event):
        assert event is not None
        self.deps.update(event)

    def wait_stream(self, stream):
        self.deps.update(stream.deps)


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.main, self.side = Stream(), Stream()
        self.current = self.main
        self.operations = {}
        self.outputs = {}

        @contextmanager
        def switch(stream, enabled):
            previous = self.current
            if enabled:
                self.current = stream
            try:
                yield
            finally:
                self.current = previous

        def op(name, value, *inputs):
            for x in inputs:
                assert x.deps <= self.current.deps, (name, x.deps, self.current.deps)
            self.operations[name] = set(self.current.deps)
            self.current.deps.add(name)
            result = Tensor(value, self.current.deps)
            self.outputs[name] = result
            return result

        def matmul(x, weight, scale, **kwargs):
            name = "gateup" if weight == 2 else "down"
            return op(name, x.value * weight, x)

        def activation(**kw):
            x = kw["x"]
            value = x.value / (1 + math.exp(-x.value * kw["glu_alpha"])) + kw["glu_bias"]
            return op("swiglu", value, x), Tensor(1)

        quant = NS(W8A8=1, W4A8=2, W8A8MXFP=3, W4A8MXFP=4)
        self.context = NS(moe_comm_type="fused", use_mega_moe=False)
        self.config = NS(enable_fused_mc2=1)
        self.ns = dict(
            torch=NS(
                int8="int8",
                int32="int32",
                npu=NS(current_stream=lambda: self.current),
                ops=NS(_C_ascend=NS(npu_dequant_swiglu_quant=activation)),
            ),
            torch_npu=NS(npu_dynamic_quant=lambda x: (op("quant", x.value, x), Tensor(1)), npu_quant_matmul=matmul),
            F=NS(linear=lambda x, w: op("gating", x.value * w, x)),
            npu_stream_switch=switch,
            shared_experts_calculation_stream=lambda: self.side,
            QuantType=quant,
            AscendSituAndMul=type("Situ", (), {}),
            _EXTRA_CTX=self.context,
            MoECommType=NS(FUSED_MC2="fused"),
            get_ascend_config=lambda: self.config,
            logger=NS(info_once=lambda *a: None, warning_once=lambda *a: None),
        )
        path = "vllm_ascend/ops/fused_moe/fused_moe.py"
        self.schedule = load_method(ROOT, path, "AscendMoERunner", "_scheduled_fused_mc2_shared", self.ns)
        self.original = load_method(ROOT, path, "AscendMoERunner", "_forward_shared_experts", self.ns)
        self.guard = load_method(ROOT, path, "AscendMoERunner", "_can_schedule_fused_mc2_shared", self.ns)
        self.runner = NS(
            fused_mc2_shared_schedule=True,
            is_internal_router=True,
            activation="silu",
            quant_type=1,
            multistream_overlap_shared_expert=False,
            gate=NS(weight_fp32=5),
            routed_experts=NS(swiglu_limit=0, swiglu_alpha=0.8, swiglu_beta=0.2),
            _shared_experts=NS(
                gate_up_proj=NS(weight=2, weight_scale=1, weight_scale_fp32=1),
                down_proj=NS(weight=3, weight_scale=1),
                act_fn=object(),
            ),
            _prepare_shared_expert_input=lambda x: x,
            _finalize_shared_expert_output=lambda x: op("finalize", x.value, x),
            _quant_method=NS(quant_method=type("AscendW8A8DynamicFusedMoEMethod", (), {})()),
        )

        def routed(x, logits, before_fused_experts):
            op("topk", logits.value, logits)
            before_fused_experts()
            return op("dispatch", x.value + logits.value, x)

        self.runner.no_shared_forward_impl = routed

    def test_schedule_dependencies_and_baseline_math(self):
        x = Tensor(0.3)
        shared, routed = self.schedule(self.runner, x, x)
        deps = dict(self.operations)
        self.assertIn("gateup", deps["gating"])
        self.assertNotIn("gating", deps["swiglu"])
        self.assertIn("gating", deps["down"])
        self.assertNotIn("down", deps["topk"])
        self.assertIn("down", deps["dispatch"])
        self.assertIn("topk", deps["dispatch"])
        self.assertIn("down", deps["finalize"])
        self.assertIn(self.side, x.streams)
        self.assertIn(self.main, self.outputs["down"].streams)
        self.assertAlmostEqual(routed.value, 1.8)
        events = NS(
            before_routed_experts=frozenset(),
            after_routed_experts=None,
            before_gmm2=None,
            before_combine=None,
            swiglu_limit=0,
            swiglu_alpha=0.8,
            swiglu_beta=0.2,
        )
        baseline = self.original(self.runner, x, events)
        self.assertAlmostEqual(shared.value, baseline.value)

    def test_w4a8_uses_baseline_default_activation_parameters(self):
        self.runner.quant_type = 2
        x = Tensor(0.3)
        shared, _ = self.schedule(self.runner, x, x)
        events = NS(
            before_routed_experts=frozenset(),
            after_routed_experts=None,
            before_gmm2=None,
            before_combine=None,
            swiglu_limit=0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
        )
        self.assertAlmostEqual(shared.value, self.original(self.runner, x, events).value)

    def test_none_activation_parameters_match_defaults(self):
        self.runner.routed_experts = NS(swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None)
        x = Tensor(0.3)
        shared, _ = self.schedule(self.runner, x, x)
        events = NS(
            before_routed_experts=frozenset(),
            after_routed_experts=None,
            before_gmm2=None,
            before_combine=None,
            swiglu_limit=0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
        )
        self.assertAlmostEqual(shared.value, self.original(self.runner, x, events).value)

    def test_real_adapter_and_quant_methods_call_hook_after_topk(self):
        for filename, classname in (
            ("w8a8_dynamic.py", "AscendW8A8DynamicFusedMoEMethod"),
            ("w4a8.py", "AscendW4A8DynamicFusedMoEMethod"),
        ):
            calls = []
            weights = NS()
            weights.to = lambda dtype, weights=weights: weights
            indices = object()
            self.ns.update(
                get_moe_num_logical_experts=lambda *a, **kw: 4,
                select_experts=lambda calls=calls, weights=weights, indices=indices, **kw: (
                    calls.append("topk") or weights,
                    indices,
                ),
                build_fused_experts_input=lambda **kw: kw,
            )
            self.ns["torch"].float32 = "fp32"
            self.ns["torch"].tensor = lambda *a, **kw: object()
            self.context.moe_comm_method = NS(fused_experts=lambda calls=calls, **kw: calls.append("dispatch") or 42)
            method = load_method(ROOT, "vllm_ascend/quantization/methods/" + filename, classname, "apply", self.ns)
            scheme = NS(dynamic_eplb=False, in_dtype="bf16", quant_type=1, is_per_channel_weight=True)
            scheme.apply = lambda method=method, scheme=scheme, **kw: method(scheme, **kw)
            adapter_method = load_method(
                ROOT, "vllm_ascend/quantization/method_adapters.py", "AscendFusedMoEMethod", "apply", self.ns
            )
            adapter = NS(quant_method=scheme, tid2eid=None)
            layer = NS(
                w13_weight=1,
                w2_weight=2,
                fused_w1_scale=1,
                fused_w2_scale=1,
                w13_weight_scale=1,
                w2_weight_scale=1,
                swiglu_limit=0,
            )
            result = adapter_method(
                adapter,
                layer,
                Tensor(1),
                NS(shape=(6, 4)),
                2,
                True,
                before_fused_experts=lambda calls=calls: calls.append("wait_down"),
            )
            self.assertEqual(result, 42)
            self.assertEqual(calls, ["topk", "wait_down", "dispatch"])
            calls.clear()
            self.assertEqual(adapter_method(adapter, layer, Tensor(1), NS(shape=(6, 4)), 2, True), 42)
            self.assertEqual(calls, ["topk", "dispatch"])

    def test_supported_guard_and_shared_layout_rejections(self):
        for projection in (self.runner._shared_experts.gate_up_proj, self.runner._shared_experts.down_proj):
            projection.weight = NS(dtype="int8")
            projection.weight_scale = NS(ndim=1)
        self.assertTrue(self.guard(self.runner))
        self.runner._shared_experts.down_proj.weight_scale.ndim = 2
        self.assertFalse(self.guard(self.runner))
        self.runner._shared_experts.down_proj.weight_scale.ndim = 1
        self.runner._shared_experts.down_proj.bias = 1
        self.assertFalse(self.guard(self.runner))
        self.runner._shared_experts.down_proj.bias = None
        self.runner._shared_experts.expert_gate = object()
        self.assertFalse(self.guard(self.runner))

    def test_unsupported_paths_fall_back(self):
        for name, value in (
            ("fused_mc2_shared_schedule", False),
            ("is_internal_router", False),
            ("quant_type", 99),
            ("activation", "gelu"),
        ):
            old = getattr(self.runner, name)
            setattr(self.runner, name, value)
            self.assertFalse(self.guard(self.runner))
            setattr(self.runner, name, old)
        self.context.use_mega_moe = True
        self.assertFalse(self.guard(self.runner))
        self.context.use_mega_moe = False
        self.context.moe_comm_type = "allgather"
        self.assertFalse(self.guard(self.runner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root
    unittest.main(argv=[__file__, *remaining])
