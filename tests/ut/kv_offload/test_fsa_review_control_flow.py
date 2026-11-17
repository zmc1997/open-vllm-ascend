# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for the FSA configuration and scheduling paths."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

_ROOT = Path(__file__).resolve().parents[3]
_MANAGER = "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py"
_SFA = "vllm_ascend/attention/sfa_kv_offload.py"
_SCHEDULER = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"


def _methods(path, owner, names, namespace):
    source = _ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


class TestFSAReviewControlFlow(unittest.TestCase):
    def test_native_fused_writeback_uses_the_capture_flag_seen_by_consumer(self):
        context = NS(capturing=False, cudagraph_runtime_mode="piecewise")
        manager = Mock()
        scope = _methods(
            _SFA,
            "AscendSFAKVOffloadImpl",
            {"exec_kv", "_in_graph_runtime"},
            {
                "get_forward_context": lambda: context,
                "is_forward_context_available": lambda: True,
                "get_sparse_kv_offload_manager": lambda: manager,
                "CUDAGraphMode": NS(NONE="none"),
            },
        )
        layer = NS(
            _is_decode_only=lambda _: True,
            _compute_kv_only=lambda *_: ("key", "rope"),
            _offload_layer_name=lambda: "layer",
            _cpu_cache_pair=lambda *_: ("host_key", "host_rope"),
            _in_graph_runtime=scope["_in_graph_runtime"],
        )
        for mode, capturing, fused, expected in (
            ("piecewise", False, True, False),
            ("full", True, True, True),
            ("none", False, True, False),
            ("piecewise", False, False, True),
        ):
            with self.subTest(mode=mode, capturing=capturing, fused=fused):
                context.capturing, context.cudagraph_runtime_mode = capturing, mode
                layer.use_fused_overlap = fused
                self.assertEqual(scope["exec_kv"](layer, None, None, None, None, None, None), ("rope", "key"))
                self.assertIs(manager.offload_new_kv.call_args.kwargs["capturing"], expected)

    def test_eager_peer_notification_only_when_reusing_a_multi_token_plan(self):
        stream = Mock()
        scatter = Mock()
        method = _methods(
            _MANAGER,
            "SparseKVOffloadManager",
            {"inject_current_kv_into_selection"},
            {"torch_npu": NS(npu=NS(current_stream=lambda: stream), npu_scatter_nd_update_=scatter)},
        )["inject_current_kv_into_selection"]
        buffer = MagicMock()
        buffer.shape = (4, 8)
        manager = NS(
            tp_rank=0,
            tp_size=2,
            tp_group=Mock(),
            _get_offload_layer_id=lambda name: int(name),
            current_kv_by_layer={i: (buffer, buffer) for i in (0, 1)},
            fused_async_plan=False,
            fused_plan_current_linear_slots_npu=buffer,
            producer_writeback_layers=set(),
            fused_overlap_plan_owner_layer_id=0,
            fused_plan_status_npu=object(),
        )
        for owner, multi_token in ((True, True), (False, False), (False, True)):
            with self.subTest(owner=owner, multi_token=multi_token):
                manager.tp_group.reset_mock()
                method(manager, "0" if owner else "1", 4, buffer, buffer, multi_token_decode=multi_token)
                self.assertEqual(manager.tp_group.broadcast.call_count, int(not owner and multi_token))
        stream.wait_stream.assert_not_called()

    def test_store_port_base_is_configurable_and_valid_for_the_dp_index(self):
        path = _ROOT / "vllm_ascend/ascend_config.py"
        cls = next(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == "SparseKVOffloadConfig"
        )
        scope = {"Any": object}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
        config = scope["SparseKVOffloadConfig"]
        vllm_config = NS(
            model_config=NS(hf_text_config=NS(index_topk=2048)),
            parallel_config=NS(
                data_parallel_index=8,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            kv_transfer_config=NS(is_kv_consumer=True),
            use_v2_model_runner=False,
        )
        self.assertEqual(config(vllm_config, {"enabled": True}).store_port_base, 8500)
        self.assertEqual(config(vllm_config, {"enabled": True, "store_port_base": 8800}).store_port_base, 8800)
        for port in (0, 65528):
            with self.subTest(port=port), self.assertRaisesRegex(ValueError, "valid TCP port"):
                config(vllm_config, {"enabled": True, "store_port_base": port})

    def test_shared_eagle_discards_tail_but_keeps_store_hit_extent(self):
        methods = _methods(
            _SCHEDULER,
            "KVPoolScheduler",
            {"get_num_new_matched_tokens", "_floor_to_cache_transfer_granularity"},
            {"LoadSpec": NS, "logger": Mock()},
        )
        scheduler = NS(
            restores_shared_prefix=True,
            use_layerwise=True,
            use_eagle=True,
            use_gva_layerwise=True,
            kv_role="kv_producer",
            load_specs={},
            load_async=True,
            lcm_block_size=128,
            cache_transfer_granularity=128,
            retention_interval=None,
            _get_layerwise_gva_hit_tokens=lambda *_: 384,
        )
        scheduler._floor_to_cache_transfer_granularity = lambda tokens: (
            methods["_floor_to_cache_transfer_granularity"](scheduler, tokens)
        )
        request = NS(request_id="prefix", num_tokens=1024, prompt_token_ids=range(1024))
        for local_hit in (256, 384, 512):
            with self.subTest(local_hit=local_hit):
                self.assertEqual(methods["get_num_new_matched_tokens"](scheduler, request, local_hit), (0, False))
                spec = scheduler.load_specs["prefix"]
                self.assertEqual(spec.vllm_cached_tokens, 256)
                self.assertEqual(spec.kvpool_cached_tokens, 256)
                self.assertEqual(spec.kvpool_store_skip_tokens, 384)


if __name__ == "__main__":
    unittest.main()
