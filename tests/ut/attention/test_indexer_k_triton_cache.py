# SPDX-License-Identifier: Apache-2.0
"""CPU-only policy and actual cache-callsite checks; NPU gates are separate."""

import argparse
import ast
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
KERNEL = "vllm_ascend.ops.triton.indexer_cache"


class Tensor:
    def __init__(self, shape, dtype="i8", *, contiguous=True, device="npu", device_index=0, owner=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = SimpleNamespace(type=device, index=device_index)
        self.contiguous, self.owner = contiguous, owner if owner is not None else self

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        if -1 in shape:
            shape = tuple(self.numel() // -math.prod(shape) if value == -1 else value for value in shape)
        return Tensor(
            shape,
            self.dtype,
            contiguous=self.contiguous,
            device=self.device.type,
            device_index=self.device.index,
            owner=self.owner,
        )


def load_sfa(namespace):
    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_init_indexer_k_triton_cache", "_can_use_indexer_k_triton_cache")
    ]
    # Execute the actual production cache-write block and its following notify.
    outer = next(
        node
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
        for node in method.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "kv_cache is not None and self.has_indexer"
    )
    method = next(method for method in cls.body if isinstance(method, ast.FunctionDef) and outer in method.body)
    index = method.body.index(outer)
    write = ast.parse("def write_cache(self, kv_cache, slot_mapping, k_li, k_li_scale, attn_metadata): pass").body[0]
    write.body = method.body[index : index + 2]
    selected = ast.ClassDef(name="SFA", bases=[], keywords=[], decorator_list=[], body=methods + [write])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["SFA"]


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        self.enabled, self.events = False, []
        self.k = MagicMock(side_effect=lambda *args: self.events.append(("triton_k", args)))
        self.scale = MagicMock(side_effect=lambda *args: self.events.append(("triton_scale", args)))
        kernel = ModuleType(KERNEL)
        kernel.scatter_k_cache, kernel.scatter_scale_cache = self.k, self.scale
        utils = ModuleType("vllm.model_executor.models.utils")
        utils.extract_layer_index = lambda prefix: int(prefix.split(".")[2])
        self.enterContext(patch.dict(sys.modules, {KERNEL: kernel, utils.__name__: utils}))
        self.namespace = {
            "torch": SimpleNamespace(
                int8="i8",
                float16="f16",
                int32="i32",
                int64="i64",
                ops=SimpleNamespace(
                    _C_ascend=SimpleNamespace(store_kv_block=lambda *args: self.events.append(("store", args)))
                ),
            ),
            "torch_npu": SimpleNamespace(npu_scatter_nd_update_=lambda *args: self.events.append(("cann", args))),
            "get_ascend_config": lambda: SimpleNamespace(enable_indexer_k_triton_cache=self.enabled),
            "HAS_TRITON": True,
            "notify_kv_cache_written": lambda name: self.events.append(("notify", name)),
        }
        self.cls = load_sfa(self.namespace)

    def layer(self, index=0, **changes):
        layer = self.cls()
        layer.is_kv_consumer, layer.tp_size, layer.has_indexer = True, 2, True
        layer.enable_sparse_li_c8, layer.enable_sparse_sfa_c8 = True, False
        layer.c8_k_cache_dtype, layer.c8_k_scale_cache_dtype, layer.head_dim = "i8", "f16", 128
        layer.o_proj = SimpleNamespace(prefix=f"model.layers.{index}.self_attn.o_proj")
        layer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78))
        )
        layer.kv_cache_indexer_k_idx, layer.kv_cache_indexer_scale_idx = 2, 3
        layer.layer_name = f"layer{index}"
        layer._use_li_c8_reshape_optim = lambda: False
        for name, value in changes.items():
            setattr(layer, name, value)
        layer._init_indexer_k_triton_cache()
        return layer

    @staticmethod
    def inputs(tokens=6, scale=True):
        return (
            [
                Tensor((8806, 128, 1, 512)),
                Tensor((8806, 128, 1, 64)),
                Tensor((8806, 128, 1, 128)),
                Tensor((8806, 128, 1, 1), "f16"),
            ],
            Tensor((tokens,), "i32"),
            Tensor((tokens, 128)),
            Tensor((tokens, 1), "f16") if scale else None,
            SimpleNamespace(group_len="length", group_key_idx="index", group_key_cache_idx="cache", block_size=128),
        )

    def test_disabled_retains_original_write_without_scale_cache(self):
        layer = self.layer(enable_sparse_li_c8=False)
        self.assertIsNone(layer._indexer_k_triton_cache)
        values = list(self.inputs(scale=False))
        values[0] = values[0][:3]
        layer.write_cache(*values)
        self.assertEqual([event[0] for event in self.events], ["cann", "notify"])

    def test_initialization_needs_only_feature_flag(self):
        self.enabled = True
        bare = self.cls()
        bare._init_indexer_k_triton_cache()
        self.assertEqual(bare._indexer_k_triton_cache, (self.k, self.scale))

    def test_all_models_roles_and_tp_use_same_dispatch(self):
        self.enabled = True
        for index, changes in (
            (78, {}),
            (79, {}),
            (0, {"is_kv_consumer": False}),
            (0, {"tp_size": 4}),
            (0, {"head_dim": 64}),
            (0, {"c8_k_cache_dtype": "fp8", "c8_k_scale_cache_dtype": "f32"}),
        ):
            with self.subTest(index=index, changes=changes):
                layer = self.layer(index, **changes)
                layer.vllm_config.model_config.hf_config.model_type = "other"
                self.assertEqual(layer._indexer_k_triton_cache, (self.k, self.scale))

    def test_actual_two_writes_preserve_alias_order_shape_and_notify(self):
        self.enabled = True
        for index, tokens, width, dtype, scale_dtype in (
            (0, 6, 128, "i8", "f16"),
            (78, 1, 64, "i8", "f32"),
            (79, 17, 97, "fp8", "f32"),
            (0, 128, 256, "bf16", "bf16"),
        ):
            self.events.clear()
            layer = self.layer(index)
            caches = [
                Tensor((31, 2)),
                Tensor((31, 2)),
                Tensor((31, 2, 1, width), dtype, contiguous=False),
                Tensor((31, 2, 1, 3), scale_dtype, contiguous=False),
            ]
            slots = Tensor((tokens,), "i64", contiguous=False)
            k = Tensor((tokens, width), dtype, contiguous=False)
            scales = Tensor((tokens, 3), scale_dtype, contiguous=False)
            layer.write_cache(caches, slots, k, scales, None)
            self.assertEqual([e[0] for e in self.events], ["triton_k", "triton_scale", "notify"])
            for event, cache, values, output_width in (
                (self.events[0], caches[2], k, width),
                (self.events[1], caches[3], scales, 3),
            ):
                self.assertIs(event[1][0].owner, cache)
                self.assertIs(event[1][1].owner, slots)
                self.assertIs(event[1][2].owner, values)
                self.assertEqual(event[1][0].shape, (62, output_width))
                self.assertEqual(event[1][2].shape, (tokens, output_width))

    def test_no_scale_or_no_indexer_and_original_block_store_semantics(self):
        self.enabled = True
        for indexer, c8, scale, reshape, expected in (
            (True, False, False, False, ["triton_k", "notify"]),
            (True, True, False, False, ["triton_k", "notify"]),
            (False, False, False, False, ["notify"]),
            (True, True, True, True, ["store", "store", "notify"]),
        ):
            self.events.clear()
            layer = self.layer(78, has_indexer=indexer, enable_sparse_li_c8=c8)
            values = list(self.inputs(17, scale))
            if not c8:
                values[0] = values[0][:3]
            layer._use_li_c8_reshape_optim = lambda reshape=reshape: reshape
            layer.write_cache(*values)
            self.assertEqual([e[0] for e in self.events], expected)

    def test_enabled_failure_does_not_fall_back_or_notify_unwritten_cache(self):
        self.enabled = True
        layer = self.layer(78)
        self.k.side_effect = RuntimeError("Triton cache failure")
        with self.assertRaisesRegex(RuntimeError, "Triton cache failure"):
            layer.write_cache(*self.inputs(tokens=1))
        self.scale.assert_not_called()
        self.assertEqual(self.events, [])

    def test_config_default_false_strict_bool(self):
        source = ROOT / "vllm_ascend/ascend_config.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        index = next(
            i
            for i, node in enumerate(init.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "enable_indexer_k_triton_cache"
                for target in node.targets
            )
        )
        code = compile(ast.Module(body=init.body[index : index + 2], type_ignores=[]), str(source), "exec")
        for config, expected in (
            ({}, False),
            ({"enable_indexer_k_triton_cache": False}, False),
            ({"enable_indexer_k_triton_cache": True}, True),
        ):
            owner = SimpleNamespace()
            exec(code, {"self": owner, "additional_config": config})
            self.assertIs(owner.enable_indexer_k_triton_cache, expected)
        for value in (0, 1, None, "true"):
            with self.assertRaises(ValueError):
                exec(code, {"self": SimpleNamespace(), "additional_config": {"enable_indexer_k_triton_cache": value}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
