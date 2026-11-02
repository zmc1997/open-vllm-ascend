# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for the FSA configuration and scheduling paths."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

_ROOT = Path(__file__).resolve().parents[3]


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


if __name__ == "__main__":
    unittest.main()
