# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for the FSA configuration and scheduling paths."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

_ROOT = Path(__file__).resolve().parents[3]
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
