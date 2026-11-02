"""Exercise production prefetch state and hooks using CPU-only device mocks."""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

_ROOT = Path(__file__).resolve().parents[3]


def _load(path, names, namespace):
    filename = _ROOT / path
    tree = ast.parse(filename.read_text(encoding="utf8"))
    nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace


class TestWeightPrefetchControlFlow(unittest.TestCase):
    def _plans(self):
        calls = []
        current = SimpleNamespace(name="main")
        aux = SimpleNamespace(name="aux", wait_event=lambda event: calls.append(("wait_aux", event)))

        class Event:
            def record(self, stream):
                calls.append(("record", self, stream))

        @contextmanager
        def stream(target):
            nonlocal current
            previous, current = current, target
            try:
                yield
            finally:
                current = previous

        current.wait_event = lambda event: calls.append(("wait_main", event))
        context = SimpleNamespace(
            is_draft_model=False,
            in_profile_run=False,
            capturing=False,
            attn_metadata={"layer": SimpleNamespace(attn_state="decode", num_prefills=0)},
        )
        namespace = _load(
            "vllm_ascend/weight_prefetch.py",
            {"DirectWeightPrefetchPlan", "OProjWeightPrefetchPlan", "MTPRotWeightPrefetchPlan"},
            {
                "torch": SimpleNamespace(int8="int8"),
                "torch_npu": SimpleNamespace(
                    npu=SimpleNamespace(Event=Event, current_stream=lambda: current, stream=stream),
                    npu_prefetch=lambda *args: calls.append(("prefetch", *args)),
                ),
                "is_forward_context_available": lambda: True,
                "get_forward_context": lambda: context,
                "AscendAttentionState": SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec"),
                "MLAPO_MAX_SUPPORTED_TOKENS": 1024,
            },
        )
        return namespace, context, aux, calls

    @staticmethod
    def _tensor(calls, *, dtype="bf16", rows=6, size=4096):
        return SimpleNamespace(
            dtype=dtype,
            shape=(rows, 128),
            device="npu:0",
            untyped_storage=lambda: SimpleNamespace(nbytes=lambda: size),
            record_stream=lambda stream: calls.append(("record_stream", stream)),
        )

    def test_config_defaults_limits_and_invalid_values(self):
        config = _load("vllm_ascend/ascend_config.py", {"WeightPrefetchConfig"}, {})["WeightPrefetchConfig"]
        self.assertFalse(config({}).enabled)
        self.assertEqual(config({}).wd_qkv_max_prefetch_bytes, 16 * 1024 * 1024)
        self.assertEqual(config({"max_prefetch_bytes": 19}).wd_qkv_max_prefetch_bytes, 19)
        fields = (
            "wd_qkv_max_prefetch_bytes",
            "wu_q_max_prefetch_bytes",
            "w_uk_t_max_prefetch_bytes",
            "o_proj_max_prefetch_bytes",
            "mtp_rot_max_prefetch_bytes",
        )
        for field in fields:
            with self.subTest(field=field):
                self.assertEqual(getattr(config({field: 23}), field), 23)
                for invalid in (-1, True, 1.5):
                    with self.assertRaises(ValueError):
                        config({field: invalid})
        for invalid, error in (
            ([], TypeError),
            ({"unknown": 0}, ValueError),
            ({"enabled": 1}, TypeError),
            ({"max_prefetch_bytes": 1, "wd_qkv_max_prefetch_bytes": 2}, ValueError),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(error):
                config(invalid)

    def test_direct_completion_and_capture_lifetime(self):
        ns, context, aux, calls = self._plans()
        weight, dependency = self._tensor(calls), self._tensor(calls)
        plan = ns["DirectWeightPrefetchPlan"](weight, "layer", aux, 8192)
        plan.submit(dependency)
        plan.wait()
        self.assertEqual(
            [call[0] for call in calls],
            [
                "record",
                "wait_aux",
                "prefetch",
                "record",
                "record_stream",
                "wait_main",
            ],
        )
        self.assertEqual(next(call for call in calls if call[0] == "prefetch"), ("prefetch", weight, dependency, 4096))
        count = len(calls)
        plan.wait()
        self.assertEqual(len(calls), count)
        context.capturing = True
        calls.clear()
        plan.submit(dependency)
        plan.wait()
        self.assertNotIn("record_stream", [call[0] for call in calls])
        context.capturing = False
        for field in ("is_draft_model", "in_profile_run"):
            setattr(context, field, True)
            calls.clear()
            plan.submit(dependency)
            plan.wait()
            self.assertEqual(calls, [])
            setattr(context, field, False)
        context.attn_metadata["layer"].num_prefills = 1
        calls.clear()
        plan.submit(dependency)
        plan.wait()
        self.assertEqual(calls, [])

    def test_oproj_hook_wraps_cann_and_triton_quantization(self):
        for use_triton, use_triton_bmm in ((False, False), (False, True), (True, False), (True, True)):
            ns, _, aux, calls = self._plans()
            plan = ns["OProjWeightPrefetchPlan"](self._tensor(calls), "layer", aux, 1024)
            result = object()

            def quantize(*_args, calls=calls, use_triton=use_triton):
                calls.append(("quantize", use_triton))
                return SimpleNamespace(dtype="int8")

            def matmul(*_args, calls=calls, result=result, **_kwargs):
                calls.append(("matmul",))
                return result

            apply = _load(
                "vllm_ascend/quantization/methods/w8a8_static.py",
                {"apply"},
                {
                    "torch": SimpleNamespace(int8="int8", ops=SimpleNamespace(vllm=SimpleNamespace(quantize=quantize))),
                    "torch_npu": SimpleNamespace(npu_quant_matmul=matmul),
                    "COMPRESSED_TENSORS_METHOD": "compressed",
                },
            )["apply"]
            layer = SimpleNamespace(
                _o_proj_weight_prefetch=plan,
                _static_triton_quant=quantize if use_triton else None,
                _static_triton_reciprocal=object(),
                _static_triton_offset=object(),
                aclnn_input_scale=object(),
                aclnn_input_scale_reciprocal=object(),
                aclnn_input_offset=object(),
                weight=object(),
                deq_scale=object(),
                quant_bias=object(),
                params_dtype="bf16",
            )
            x = self._tensor(calls)
            x.view = lambda *_args, x=x: x
            x.reshape = lambda *_args, x=x: x

            def bmm(*_args, calls=calls, x=x, **_kwargs):
                calls.append(("vup_bmm",))
                return x

            sfa = SimpleNamespace(
                _o_proj_weight_prefetch=plan,
                _vup_triton_bmm=bmm if use_triton_bmm else None,
                W_UV=SimpleNamespace(shape=(1, 128, 128)),
                local_num_heads=1,
                kv_lora_rank=128,
                v_head_dim=128,
            )
            v_up = _load(
                "vllm_ascend/attention/sfa_v1.py",
                {"_v_up_proj"},
                {"torch_npu": SimpleNamespace(npu_transpose_batchmatmul=bmm)},
            )["_v_up_proj"]
            self.assertIs(apply(None, layer, v_up(sfa, x)), result)
            stages = [call[0] for call in calls]
            self.assertEqual(stages.count("prefetch"), 1)
            self.assertLess(stages.index("prefetch"), stages.index("vup_bmm"))
            self.assertLess(stages.index("vup_bmm"), stages.index("quantize"))
            self.assertLess(stages.index("quantize"), stages.index("wait_main"))
            self.assertLess(stages.index("wait_main"), stages.index("matmul"))

    def test_oproj_does_not_prefetch_without_quantization_or_under_mmrs(self):
        ns, context, aux, calls = self._plans()
        plan = ns["OProjWeightPrefetchPlan"](self._tensor(calls), "layer", aux, 1024)
        plan.submit(self._tensor(calls, dtype="int8"))
        plan.wait()
        context.mmrs_fusion = True
        plan.submit(self._tensor(calls))
        plan.wait()
        self.assertEqual(calls, [])

    def test_mtp_logits_prefetch_skips_last_step_and_prefill(self):
        ns, context, aux, calls = self._plans()
        context.is_draft_model = True
        logits = self._tensor(calls)
        plan = ns["MTPRotWeightPrefetchPlan"](self._tensor(calls), aux, 1024)
        proposer = SimpleNamespace(
            model=SimpleNamespace(compute_logits=Mock(return_value=logits)),
            _mtp_rot_weight_prefetch=plan,
            method="mtp",
            parallel_drafting=False,
        )
        compute = _load("vllm_ascend/spec_decode/llm_base_proposer.py", {"_compute_draft_logits"}, {})[
            "_compute_draft_logits"
        ]
        for total in (1, 2, 5):
            calls.clear()
            for step in range(total):
                self.assertIs(compute(proposer, object(), has_next_draft=step + 1 < total, is_prefill=False), logits)
                plan.wait()
            self.assertEqual(sum(call[0] == "prefetch" for call in calls), total - 1)
        for changes in ({"is_prefill": True}, {"is_prefill": None}):
            calls.clear()
            compute(proposer, object(), has_next_draft=True, **changes)
            plan.wait()
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
