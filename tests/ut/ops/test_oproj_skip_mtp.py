# SPDX-License-Identifier: Apache-2.0
"""Run CPU role/rank contracts directly with Python; no torch or NPU needed."""

from __future__ import annotations

import argparse
import ast
import copy
import logging
import re
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    definitions = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(node for node in body if getattr(node, "name", None) == part)
            body = getattr(node, "body", [])
        definitions.append(copy.deepcopy(node))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)


def model_config(main_layers=7, mtp_layers=2):
    text = types.SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=main_layers)
    draft = types.SimpleNamespace(model_type="deepseek_mtp", num_hidden_layers=main_layers,
                                  num_nextn_predict_layers=mtp_layers)
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=text, hf_config=text, is_moe=True, enforce_eager=False),
        speculative_config=types.SimpleNamespace(method="mtp", draft_model_config=types.SimpleNamespace(hf_config=draft)),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=2, data_parallel_size=16,
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        kv_transfer_config=types.SimpleNamespace(is_kv_consumer=True), additional_config={},
    )


class SkipMTPContract(unittest.TestCase):
    def setUp(self):
        self.tp = types.SimpleNamespace(rank_in_group=0, world_size=2)
        self.otp = types.SimpleNamespace(rank_in_group=0, world_size=2)
        self.ascend = types.SimpleNamespace()
        self.ns = {
            "logger": logging.getLogger(__name__), "get_ascend_config": lambda: self.ascend,
            "get_tp_group": lambda: self.tp, "get_otp_group": lambda: self.otp,
            "get_tensor_model_parallel_world_size": lambda: self.tp.world_size,
            "get_tensor_model_parallel_rank": lambda: self.tp.rank_in_group,
            "enable_sp": lambda *args: False, "mlp_tp_enable": lambda: False,
            "shared_expert_dp_enabled": lambda: False,
            "extract_layer_index": lambda prefix: int(re.search(r"(?:^|\.)layers\.(\d+)\.", prefix).group(1)),
        }
        load("vllm_ascend/ascend_config.py", ["FinegrainedTPConfig"], self.ns)
        load("vllm_ascend/utils.py", ["oproj_tp_enable"], self.ns)
        load("vllm_ascend/ops/linear_op.py",
             ["CustomLinearOp", "CustomRowParallelOp", "OProjRowParallelOp", "_oproj_tp_enabled_for_layer",
              "_get_row_parallel_op", "get_parallel_op"], self.ns)
        load("vllm_ascend/attention/sfa_v1.py", ["AscendSFAImpl.forward"], self.ns)
        self.sfa_forward = self.ns.pop("forward")
        self.configure(True)

    def configure(self, enabled, config=None):
        values = {"oproj_tensor_parallel_size": 4, "oproj_skip_mtp": enabled, "lmhead_tensor_parallel_size": 16}
        self.ascend.finegrained_tp_config = self.ns["FinegrainedTPConfig"](values, config or model_config())

    def select(self, index, suffix="self_attn.o_proj"):
        return self.ns["get_parallel_op"](False, f"model.layers.{index}.{suffix}", types.SimpleNamespace(), "row")

    def test_default_preserves_all_layers_tp4(self):
        fine = self.ns["FinegrainedTPConfig"]({"oproj_tensor_parallel_size": 4}, model_config())
        self.assertFalse(fine.oproj_skip_mtp)
        self.ascend.finegrained_tp_config = fine
        self.assertIsInstance(self.select(7)[0], self.ns["OProjRowParallelOp"])

    def test_roles_and_indices_select_before_weight_sharding(self):
        for dp in range(2):
            self.otp.rank_in_group = dp
            for tp in range(2):
                self.tp.rank_in_group = tp
                for index in (0, 6):
                    op, rank, size = self.select(index)
                    self.assertIsInstance(op, self.ns["OProjRowParallelOp"])
                    self.assertEqual((rank, size, 16384 // size), (tp * 2 + dp, 4, 4096))
                for index in (7, 8):
                    self.assertEqual(self.select(index), (None, tp, 2))
                    # Serialized names can include mtp_block; construction names do not.
                    self.assertEqual(self.select(index, "mtp_block.self_attn.o_proj"), (None, tp, 2))
        self.assertEqual(self.ascend.finegrained_tp_config.lmhead_tensor_parallel_size, 16)
        with self.assertRaises(ValueError):
            self.select(9)
        tree = ast.parse((ROOT / "vllm_ascend/ops/linear.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if getattr(node, "name", None) == "AscendRowParallelLinear")
        init = next(node for node in cls.body if getattr(node, "name", None) == "__init__")
        calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)]
        selected = next(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "get_parallel_op")
        allocation = next(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "create_weights")
        self.assertLess(selected, allocation)

    def test_reject_wrong_roles_or_invalid_flag(self):
        with self.assertRaises(ValueError):
            self.configure("true")
        for kind in ("not_mtp", "wrong_draft", "no_mtp_layers", "wrong_target"):
            config = model_config()
            if kind == "not_mtp":
                config.speculative_config.method = "draft_model"
            elif kind == "wrong_draft":
                config.speculative_config.draft_model_config.hf_config.model_type = "glm4_moe_mtp"
            elif kind == "no_mtp_layers":
                config.speculative_config.draft_model_config.hf_config.num_nextn_predict_layers = 0
            else:
                config.model_config.hf_text_config.model_type = "deepseek_v3"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.configure(True, config)

    def test_idle_mtp_skips_only_its_cross_dp_oproj(self):
        class Tensor:
            shape = (6, 8192)
            def new_zeros(self, shape):
                return Tensor()
            def fill_(self, value):
                self.value = value
                return self

        class Projection:
            def __init__(self, size):
                self.tp_size, self.calls = size, 0
            def __call__(self, x):
                self.calls += 1

        for size, expected_calls in ((4, 5), (2, 0)):
            projection = Projection(size)
            attention = types.SimpleNamespace(o_proj=projection, local_num_heads=32, v_head_dim=256)
            for _ in range(5):
                output = self.sfa_forward(attention, "model.layers.7.self_attn.attn", Tensor(), (), None, output=Tensor())
                self.assertEqual(output.value, 0)
            self.assertEqual(projection.calls, expected_calls)

    def test_native_communication_requires_valid_main_only_quant_pack(self):
        cls = self.ns["FinegrainedTPConfig"]
        self.assertFalse(cls({}, model_config()).oproj_native_communication)
        values = dict(oproj_tensor_parallel_size=4, oproj_skip_mtp=True, oproj_quant_pack=True,
                      oproj_communication="all_reduce", oproj_native_communication=True)
        self.assertTrue(cls(values, model_config()).oproj_native_communication)
        for name, bad in (("oproj_skip_mtp", False), ("oproj_quant_pack", False),
                          ("oproj_communication", "reduce_scatter"), ("oproj_native_communication", "true")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cls(dict(values, **{name: bad}), model_config())

    def test_mtp_quant_bias_uses_actual_tp2_rank_once(self):
        class RowLayer:
            pass

        captured = []
        self.ns.update(RowParallelLinear=RowLayer, torch=types.SimpleNamespace(int8="int8"),
                       COMPRESSED_TENSORS_METHOD="compressed-tensors",
                       torch_npu=types.SimpleNamespace(npu_quant_matmul=lambda *args, **kwargs: captured.append(kwargs["bias"])))
        load("vllm_ascend/quantization/methods/w8a8_static.py", ["AscendW8A8LinearMethod.apply"], self.ns)
        scheme = types.SimpleNamespace()
        scheme.apply = types.MethodType(self.ns.pop("apply"), scheme)
        load("vllm_ascend/quantization/method_adapters.py", ["AscendLinearMethod.apply"], self.ns)
        adapter = types.SimpleNamespace(quant_method=scheme)
        for dp in range(2):
            self.otp.rank_in_group = dp
            for tp in range(2):
                self.tp.rank_in_group = tp
                layer = RowLayer()
                layer.prefix = "model.layers.7.self_attn.o_proj"
                layer.custom_op, layer.tp_rank, layer.tp_size = self.select(7)
                layer.quant_bias, layer.weight, layer.deq_scale, layer.params_dtype = 123, "weight", "scale", "bf16"
                self.ns["apply"](adapter, layer, types.SimpleNamespace(dtype="int8"))
        self.assertEqual(captured, [123, None, 123, None])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
