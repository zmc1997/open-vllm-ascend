"""CPU checks of production routing and stream order, without device imports."""

import ast
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"
_COMPILER_SOURCE = _SOURCE.parents[1] / "compilation/compiler_interface.py"


def _method(name, namespace):
    tree = ast.parse(_SOURCE.read_text(encoding="utf8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)
    return namespace[name]


class TestSFAIndexerOverlapControlFlow(unittest.TestCase):
    def test_static_quantization_preserves_triton_and_cann_selection(self):
        class StaticScheme:
            pass

        cann, triton = Mock(return_value=object()), Mock(return_value=object())
        quantize = _method(
            "_indexer_q_quantize",
            {
                "torch": SimpleNamespace(int8="int8", ops=SimpleNamespace(vllm=SimpleNamespace(quantize=cann))),
                "AscendW8A8LinearMethod": StaticScheme,
            },
        )
        layer = SimpleNamespace(
            aclnn_input_scale=object(),
            aclnn_input_scale_reciprocal=object(),
            aclnn_input_offset=object(),
            _static_triton_reciprocal=object(),
            _static_triton_offset=object(),
        )
        impl = SimpleNamespace(wq_b=layer, _get_layer_quant_method=lambda _: StaticScheme())
        value = SimpleNamespace(dtype="bf16")
        self.assertIs(quantize(impl, value), cann.return_value)
        cann.assert_called_once_with(
            value, layer.aclnn_input_scale, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset
        )
        layer._static_triton_quant = triton
        self.assertIs(quantize(impl, value), triton.return_value)
        triton.assert_called_once_with(value, layer._static_triton_reciprocal, layer._static_triton_offset)
        for original in (SimpleNamespace(dtype="int8"), (object(), object())):
            self.assertIs(quantize(impl, original), original)
        impl._get_layer_quant_method = lambda _: object()
        self.assertIs(quantize(impl, value), value)
        self.assertEqual(cann.call_count, 1)
        self.assertEqual(triton.call_count, 1)

    def test_projection_keeps_original_layer_and_dynamic_scale_arguments(self):
        device_mm = Mock(return_value=object())
        project = _method(
            "_indexer_q_matmul",
            {
                "torch": SimpleNamespace(float8_e4m3fn="fp8"),
                "torch_npu": SimpleNamespace(npu_quant_matmul=device_mm),
            },
        )
        layer = Mock(return_value=(object(), None))
        impl = SimpleNamespace(wq_b=layer)
        value = SimpleNamespace(dtype="int8")
        self.assertIs(project(impl, value, SimpleNamespace(dtype="bf16")), layer.return_value[0])
        layer.assert_called_once_with(value)
        data = SimpleNamespace(dtype="int8", shape=(6, 2048), view=Mock())
        data.view.return_value = data
        scale = SimpleNamespace(dim=lambda: 2, shape=(6, 1), squeeze=Mock(return_value=object()))
        layer.weight, layer.weight_scale = object(), object()
        self.assertIs(project(impl, (data, scale), SimpleNamespace(dtype="bf16")), device_mm.return_value)
        device_mm.assert_called_once_with(
            data,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=scale.squeeze.return_value,
            bias=None,
            output_dtype="bf16",
        )
        self.assertEqual(layer.call_count, 1)

    def test_native_overlap_orders_streams_and_reuses_k_projection(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self._check_overlap(enabled)

    def test_native_cached_topk_skips_q_but_keeps_k_cache_updates(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self._check_overlap(enabled, skip_topk=True)

    def test_mlapo_overlap_uses_fused_q_and_preserves_cached_topk(self):
        for mode in ("mlapo", "direct"):
            for enabled in (False, True):
                for skip_topk in (False, True):
                    for speculative in (False, True):
                        with self.subTest(mode=mode, enabled=enabled, skip_topk=skip_topk, speculative=speculative):
                            self._check_overlap(enabled, skip_topk, mode=mode, speculative=speculative)

    def test_overlap_schedule_versions_compiler_cache(self):
        tree = ast.parse(_COMPILER_SOURCE.read_text(encoding="utf8"))
        method = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compute_hash"
        )
        config = SimpleNamespace(
            ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False),
            multistream_sfa_indexer_overlap=False,
            weight_prefetch_config=None,
        )
        namespace = {"get_ascend_config": lambda: config, "logger": Mock()}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(_COMPILER_SOURCE), "exec"), namespace)
        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(__version__="test")}):
            compiler = SimpleNamespace()
            disabled = namespace["compute_hash"](compiler, object())
            config.multistream_sfa_indexer_overlap = True
            enabled = namespace["compute_hash"](compiler, object())
        self.assertNotEqual(enabled, disabled)

    def test_prolog_keeps_original_indexer_order(self):
        for skip_topk in (False, True):
            with self.subTest(skip_topk=skip_topk):
                self._check_overlap(True, skip_topk, mode="prolog")

    def test_shared_indexer_layers_do_not_schedule_local_qk(self):
        for mode in ("native", "mlapo", "prolog"):
            with self.subTest(mode=mode):
                self._check_overlap(True, True, mode=mode, has_indexer=False)

    def _check_overlap(self, enabled, skip_topk=False, mode="native", speculative=False, has_indexer=True):
        class Boundary(Exception):
            pass

        native, mlapo, prolog = object(), object(), object()
        states = SimpleNamespace(DecodeOnly=object(), SpecDecoding=object())
        events = []
        main = SimpleNamespace(wait_stream=lambda _: events.append("main_wait_aux"))
        aux = SimpleNamespace(wait_stream=lambda _: events.append("aux_wait_main"))
        current_stream = Mock(return_value=main)
        auxiliary_stream = Mock(return_value=aux)
        q_c, kw = object(), object()
        hidden_states, ql_nope, q_pe = object(), object(), object()
        k_li, k_scale = Mock(shape=(6, 128)), Mock(shape=(6, 1))
        slot_mapping = Mock()
        kv_cache = tuple(Mock() for _ in range(4))
        quantized_q, q_li, topk, cached_topk = object(), object(), object(), object()
        projected = SimpleNamespace(split=lambda *_args, **_kwargs: (q_c, object()))
        impl = SimpleNamespace(
            enable_dsa_cp=False,
            enable_sp=False,
            enable_dsa_cp_with_o_proj_tp=False,
            has_indexer=has_indexer,
            preprocess_type={"native": native, "mlapo": mlapo, "direct": mlapo, "prolog": prolog}[mode],
            enable_indexer_multistream=enabled,
            skip_topk=skip_topk,
            use_index_cache=True,
            enable_sparse_li_c8=True,
            enable_sparse_sfa_c8=False,
            layer_name="layer",
            kv_cache_indexer_k_idx=2,
            kv_cache_indexer_scale_idx=3,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            _compose_sfa_kv_cache=lambda value: value,
            _get_sfa_kv_slot_mapping=lambda _: object(),
            fused_qkv_a_proj=Mock(return_value=(projected, None)),
            q_a_layernorm=Mock(side_effect=lambda x: events.append("q_norm") or x),
            wk_weights_proj=Mock(side_effect=lambda _: (events.append("k_matmul") or kw, None)),
            _indexer_q_quantize=Mock(side_effect=lambda _: events.append("q_quant") or quantized_q),
            _indexer_q_matmul=Mock(side_effect=lambda *_args: events.append("q_matmul") or q_li),
            _indexer_pre_vector_part=Mock(side_effect=lambda *_args: events.append("k_vector") or k_li),
            _indexer_pre_c8_quant=Mock(side_effect=lambda _: (events.append("k_quant") or k_li, k_scale)),
            indexer_select_pre_process=Mock(return_value=(k_li, k_scale)),
            exec_kv=Mock(return_value=(object(), object())),
            _maybe_gather_kv_for_dsacp=Mock(side_effect=lambda *args: (args[3], args[4], None, [])),
            _q_proj_and_k_up_proj=Mock(return_value=(object(), object())),
            rope_single=Mock(),
            _record_query_gather_context=Mock(),
            _maybe_store_kvcache_for_c8_n_dsacp=Mock(side_effect=lambda *args: (args[0], args[1], args[3], None, None)),
            _get_full_kv=Mock(return_value=k_li),
            _use_li_c8_reshape_optim=lambda: False,
            _indexer_k_triton_cache=(
                Mock(side_effect=lambda *_args: events.append("k_cache")),
                Mock(side_effect=lambda *_args: events.append("scale_cache")),
            ),
            _get_indexcache_topk_indices=Mock(side_effect=lambda _: events.append("reuse_topk") or cached_topk),
            indexer_select_post_process=Mock(side_effect=lambda **_kwargs: events.append("compute_topk") or topk),
            _update_indexcache_topk_indices=Mock(),
            _execute_sparse_flash_attention_process=Mock(side_effect=Boundary),
            _sfa_preprocess_mlapo=Mock(
                side_effect=lambda **_kwargs: events.append(mode) or (hidden_states, ql_nope, q_pe, q_c)
            ),
            _sfa_preprocess_prolog_v3=Mock(
                side_effect=lambda **_kwargs: events.append("prolog") or (hidden_states, ql_nope, q_pe, q_c)
            ),
        )
        stream_switch = Mock(side_effect=lambda *_args, **_kwargs: nullcontext())
        namespace = {
            "AscendAttentionState": states,
            "PreprocessType": SimpleNamespace(NATIVE=native, MLAPO=mlapo, PROLOG_V3=prolog),
            "MLAPO_MAX_SUPPORTED_TOKENS": 1024,
            "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=current_stream)),
            "os": os,
            "attention_calculation_stream": auxiliary_stream,
            "npu_stream_switch": stream_switch,
            "wait_for_kv_layer_from_connector": lambda _: events.append("kv_wait"),
            "notify_kv_cache_written": lambda _: events.append("notify"),
            "record_attention_compute_start": lambda: events.append("prefetch_gate"),
        }
        impl._indexer_qk_pre_process = Mock(
            side_effect=lambda *args, **kwargs: _method("_indexer_qk_pre_process", namespace)(impl, *args, **kwargs)
        )
        forward = _method("forward", namespace)
        metadata = SimpleNamespace(
            cos=None,
            sin=None,
            slot_mapping=slot_mapping,
            cum_query_lens=None,
            seq_lens=None,
            num_input_tokens=6,
            attn_state=states.SpecDecoding if speculative else states.DecodeOnly,
        )
        slot_mapping.numel.return_value = 6
        hidden_input = Mock(shape=(6, 16)) if mode == "prolog" else hidden_states
        with patch.dict(os.environ, {"VLLM_PATCH_TRACE": "0"}), self.assertRaises(Boundary):
            forward(impl, "layer", hidden_input, kv_cache, metadata, output=object())
        tail = (["k_cache", "scale_cache"] if has_indexer else []) + [
            "notify",
            "prefetch_gate",
            "reuse_topk" if skip_topk else "compute_topk",
        ]
        overlap = has_indexer and enabled and not skip_topk and mode != "prolog"
        if overlap:
            if mode == "native":
                prefix = ["q_norm", "aux_wait_main", "q_quant", "k_matmul"]
                impl._indexer_qk_pre_process.assert_called_once_with(hidden_states, q_c, None, None)
            else:
                prefix = ["k_matmul", "kv_wait", mode, "q_quant"]
                impl._indexer_qk_pre_process.assert_called_once_with(hidden_states, q_c, None, None, indexer_kw=kw)
            self.assertEqual(
                events,
                prefix
                + [
                    "aux_wait_main",
                    "q_matmul",
                    "k_vector",
                    "main_wait_aux",
                    "k_quant",
                ]
                + (["kv_wait"] if mode == "native" else [])
                + tail,
            )
            impl._indexer_pre_vector_part.assert_called_once_with(kw, None, None)
            impl._indexer_pre_c8_quant.assert_called_once_with(k_li)
            impl._indexer_q_quantize.assert_called_once_with(q_c)
            impl._indexer_q_matmul.assert_called_once_with(quantized_q, hidden_states)
            self.assertEqual(stream_switch.call_count, 2 if mode == "native" else 1)
            impl.indexer_select_pre_process.assert_not_called()
        else:
            k_events = ["k_matmul"] if has_indexer else []
            prefix = ["q_norm"] + k_events + ["kv_wait"] if mode == "native" else k_events + ["kv_wait", mode]
            self.assertEqual(events, prefix + tail)
            if has_indexer:
                impl.indexer_select_pre_process.assert_called_once_with(kw=kw, cos=None, sin=None)
            else:
                impl.indexer_select_pre_process.assert_not_called()
            impl._indexer_qk_pre_process.assert_not_called()
            impl._indexer_q_quantize.assert_not_called()
            impl._indexer_q_matmul.assert_not_called()
            current_stream.assert_not_called()
            auxiliary_stream.assert_not_called()
        if mode == "native":
            impl._q_proj_and_k_up_proj.assert_called_once_with(q_c)
            impl._sfa_preprocess_mlapo.assert_not_called()
            impl._sfa_preprocess_prolog_v3.assert_not_called()
        else:
            impl._q_proj_and_k_up_proj.assert_not_called()
            impl.fused_qkv_a_proj.assert_not_called()
            impl.q_a_layernorm.assert_not_called()
            fused = impl._sfa_preprocess_mlapo if mode in ("mlapo", "direct") else impl._sfa_preprocess_prolog_v3
            fused.assert_called_once()
            self.assertIs(impl._execute_sparse_flash_attention_process.call_args.args[0], ql_nope)
            self.assertIs(impl._execute_sparse_flash_attention_process.call_args.args[1], q_pe)
        if has_indexer:
            impl.wk_weights_proj.assert_called_once()
            impl._indexer_k_triton_cache[0].assert_called_once_with(
                kv_cache[2].view.return_value, slot_mapping.view.return_value, k_li.view.return_value
            )
            impl._indexer_k_triton_cache[1].assert_called_once_with(
                kv_cache[3].view.return_value, slot_mapping.view.return_value, k_scale.view.return_value
            )
        else:
            impl.wk_weights_proj.assert_not_called()
            for write in impl._indexer_k_triton_cache:
                write.assert_not_called()
        if skip_topk:
            impl._get_indexcache_topk_indices.assert_called_once_with(6)
            impl.indexer_select_post_process.assert_not_called()
            impl._update_indexcache_topk_indices.assert_not_called()
        else:
            impl._get_indexcache_topk_indices.assert_not_called()
            self.assertIs(impl.indexer_select_post_process.call_args.kwargs["kw"], kw)
            self.assertIs(impl.indexer_select_post_process.call_args.kwargs["q_li"], q_li if overlap else None)
            impl._update_indexcache_topk_indices.assert_called_once_with(topk)
        self.assertIs(
            impl._execute_sparse_flash_attention_process.call_args.args[3], cached_topk if skip_topk else topk
        )


if __name__ == "__main__":
    unittest.main()
