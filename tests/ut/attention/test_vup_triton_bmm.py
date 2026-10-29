# SPDX-License-Identifier: Apache-2.0
"""CPU-only policy and dispatch tests; NPU equality/performance are separate."""

import argparse
import ast
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
KERNEL_NAME = "vllm_ascend.ops.triton.vup_bmm"


class Tensor:
    def __init__(self, shape, *, dtype="bf16", device=None, contiguous=True, format=2):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = device or SimpleNamespace(type="npu", index=0)
        self.ndim, self.format = len(shape), format
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def view(self, *shape):
        if -1 in shape:
            shape = tuple(self.numel() // -math.prod(shape) if n == -1 else n for n in shape)
        return Tensor(shape, dtype=self.dtype, device=self.device, format=self.format)

    reshape = view


def load_sfa(namespace):
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    selected = ast.ClassDef(
        name=cls.name,
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_init_vup_triton_bmm", "_v_up_proj")
        ],
    )
    body = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("_VUP_TRITON_") for t in n.targets)
    ]
    module = ast.fix_missing_locations(ast.Module(body=body + [selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls.name]


class TestVupDispatch(unittest.TestCase):
    def setUp(self):
        self.enabled = False
        self.triton = MagicMock(side_effect=lambda x, w: Tensor((x.shape[0], w.shape[0], w.shape[2]), dtype=x.dtype))
        self.cann = MagicMock(side_effect=lambda x, w, **kw: Tensor((x.shape[0], 32, 256)))
        kernel = ModuleType(KERNEL_NAME)
        kernel.vup_bmm = self.triton
        utils = ModuleType("vllm.model_executor.models.utils")
        utils.extract_layer_index = lambda prefix: int(prefix.split(".")[2])
        self.enterContext(patch.dict(sys.modules, {KERNEL_NAME: kernel, utils.__name__: utils}))
        self.cls = load_sfa(
            {
                "torch": SimpleNamespace(bfloat16="bf16"),
                "torch_npu": SimpleNamespace(npu_transpose_batchmatmul=self.cann, get_npu_format=lambda x: x.format),
                "get_ascend_config": lambda: SimpleNamespace(enable_vup_triton_bmm=self.enabled),
                "ACL_FORMAT_FRACTAL_ND": 2,
            }
        )

    def layer(self, index=0, **changes):
        layer = self.cls()
        layer.is_kv_consumer, layer.tp_size = True, 2
        layer.local_num_heads, layer.kv_lora_rank, layer.v_head_dim = 32, 512, 256
        layer.o_proj = SimpleNamespace(prefix=f"model.layers.{index}.self_attn.o_proj")
        layer.W_UV = Tensor((32, 512, 256))
        layer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78))
        )
        for name, value in changes.items():
            setattr(layer, name, value)
        return layer

    def test_disabled_has_no_new_dispatch_or_weight_copy(self):
        layer = self.layer()
        weight = layer.W_UV
        layer._init_vup_triton_bmm()
        self.assertIsNone(layer._vup_triton_bmm)
        self.assertIs(layer.W_UV, weight)
        result = layer._v_up_proj(Tensor((6, 32, 512)))
        self.assertEqual(result.shape, (6, 8192))
        self.cann.assert_called_once()
        self.triton.assert_not_called()

    def test_enabled_dispatch_uses_actual_weights_for_all_models_and_batches(self):
        self.enabled = True
        for index, consumer, tp, shape, dtype in (
            (0, True, 2, (6, 32, 512, 256), "bf16"),
            (78, True, 4, (1, 16, 65, 127), "fp16"),
            (79, False, 16, (17, 3, 33, 19), "fp32"),
            (0, False, 1, (32, 7, 128, 64), "bf16"),
        ):
            with self.subTest(index=index, shape=shape):
                n, h, k, v = shape
                layer = self.layer(
                    index, is_kv_consumer=consumer, tp_size=tp, W_UV=Tensor((h, k, v), dtype=dtype, contiguous=False)
                )
                layer.vllm_config.model_config.hf_config.model_type = "other"
                weight = layer.W_UV
                layer._init_vup_triton_bmm()
                result = layer._v_up_proj(Tensor((n, h, k), dtype=dtype, contiguous=False))
                self.assertEqual(result.shape, (n, h * v))
                self.assertIs(self.triton.call_args.args[1], weight)
                self.assertEqual(self.triton.call_args.args[0].shape, (n, h, k))
                self.assertIs(layer.W_UV, weight)
                self.triton.assert_called_once()
                self.cann.assert_not_called()
                self.triton.reset_mock()

    def test_enabled_kernel_failure_is_visible_without_native_fallback(self):
        self.enabled = True
        layer = self.layer(78)
        layer._init_vup_triton_bmm()
        self.triton.side_effect = RuntimeError("kernel failure")
        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            layer._v_up_proj(Tensor((1, 32, 512)))
        self.cann.assert_not_called()

    def test_initialization_needs_only_feature_flag(self):
        self.enabled = True
        layer = self.cls()
        layer._init_vup_triton_bmm()
        self.assertIs(layer._vup_triton_bmm, self.triton)

    def test_config_flag_is_default_false_and_strict_bool(self):
        path = ROOT / "vllm_ascend/ascend_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(
            i
            for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "enable_vup_triton_bmm" for t in n.targets)
        )
        body = ast.Module(body=init.body[start : start + 2], type_ignores=[])
        code = compile(ast.fix_missing_locations(body), str(path), "exec")
        cases = (({}, False), ({"enable_vup_triton_bmm": False}, False), ({"enable_vup_triton_bmm": True}, True))
        for cfg, expected in cases:
            owner = SimpleNamespace()
            exec(code, {"self": owner, "additional_config": cfg})
            self.assertIs(owner.enable_vup_triton_bmm, expected)
        for value in (0, 1, "true", None):
            with self.assertRaises(ValueError):
                exec(code, {"self": SimpleNamespace(), "additional_config": {"enable_vup_triton_bmm": value}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
