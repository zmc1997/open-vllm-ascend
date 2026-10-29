# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch/ownership regressions; device equality and timing are separate."""

import argparse
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import test_oproj_triton_quant as quant_tests

ROOT = Path(__file__).resolve().parents[3]


class TestIndexerQQuant(unittest.TestCase):
    setUp = quant_tests.TestOProjTritonQuant.setUp

    @staticmethod
    def layer(prefix="model.layers.0.self_attn.indexer.wq_b", dtype="bfloat16", tp=2, weight_shape=(4096, 2048)):
        layer = quant_tests.TestOProjTritonQuant.layer(prefix, dtype, tp)
        layer.weight.data.shape = weight_shape
        layer.weight.data.transpose.return_value.contiguous.return_value.shape = weight_shape[::-1]
        return layer

    @staticmethod
    def input(tokens=6, width=2048, dtype="bfloat16", ndim=2, contiguous=True, device_type="npu", device_index=0):
        return quant_tests.TestOProjTritonQuant.input(
            dtype=dtype,
            ndim=ndim,
            contiguous=contiguous,
            shape=(tokens,) + (1,) * (ndim - 2) + (width,),
            device_type=device_type,
            device_index=device_index,
        )

    def prepare(self, layer=None):
        layer = layer or self.layer()
        self.method.process_weights_after_loading(layer)
        return layer

    def test_three_flags_are_independent(self):
        for oproj in (False, True):
            for indexer in (False, True):
                for bmm in (False, True):
                    self.config.enable_oproj_triton_quant = oproj
                    self.config.enable_indexer_q_triton_quant = indexer
                    self.config.enable_vup_triton_bmm = bmm
                    for prefix, shape, enabled in (
                        ("model.layers.0.self_attn.o_proj", (6, 8192), oproj),
                        ("model.layers.0.self_attn.indexer.wq_b", (6, 2048), indexer),
                        ("model.layers.0.self_attn.q_b_proj", (6, 2048), False),
                    ):
                        with self.subTest(oproj=oproj, indexer=indexer, bmm=bmm, prefix=prefix):
                            layer = self.prepare(self.layer(prefix))
                            x = self.input(*shape)
                            self.method.apply(layer, x)
                            self.assertEqual(self.triton.call_count, int(enabled))
                            self.assertEqual(self.cann.call_count, int(not enabled))
                            self.triton.reset_mock()
                            self.cann.reset_mock()

    def test_exact_indexer_prefix_only(self):
        self.config.enable_indexer_q_triton_quant = True
        for prefix in (
            "model.layers.0.self_attn.wq_b",
            "model.layers.0.self_attn.indexer.wq_b_extra",
            "model.layers.0.self_attn.indexer.wk_weights_proj",
        ):
            with self.subTest(prefix=prefix):
                layer = self.prepare(self.layer(prefix))
                self.method.apply(layer, self.input())
                self.assertIsNone(layer._static_triton_quant)
                self.cann.assert_called_once()
                self.triton.assert_not_called()
                self.cann.reset_mock()

    def test_main_and_mtp_alias_original_full_parameters(self):
        self.config.enable_indexer_q_triton_quant = True
        for index in (0, 74, 78):
            layer = self.prepare(self.layer(f"model.layers.{index}.self_attn.indexer.wq_b"))
            reciprocal = layer.aclnn_input_scale_reciprocal
            offset = layer.aclnn_input_offset
            self.assertIs(layer._static_triton_reciprocal, reciprocal)
            self.assertIs(layer._static_triton_offset, offset)
            for tokens in (1, 6, 17):
                x = self.input(tokens=tokens)
                self.method.apply(layer, x)
                self.triton.assert_called_with(x, reciprocal, offset)
            reciprocal.__getitem__.assert_not_called()
            offset.__getitem__.assert_not_called()
            reciprocal.to.assert_not_called()
            offset.to.assert_called_once_with(layer.aclnn_input_scale.dtype)
        self.cann.assert_not_called()

    def test_tp_dtype_and_weight_shape_do_not_limit_quantization(self):
        self.config.enable_indexer_q_triton_quant = True
        for dtype, tp, shape in (
            ("float16", 2, (4096, 2048)),
            ("bfloat16", 1, (4096, 2048)),
            ("bfloat16", 4, (4096, 2048)),
            ("bfloat16", 2, (4096, 1024)),
            ("bfloat16", 2, (8192, 2048)),
            ("float32", 16, (127, 129)),
        ):
            with self.subTest(dtype=dtype, tp=tp, shape=shape):
                layer = self.prepare(self.layer(dtype=dtype, tp=tp, weight_shape=shape))
                x = self.input(tokens=1, width=shape[1], dtype=dtype)
                self.method.apply(layer, x)
                self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
        self.cann.assert_not_called()

    def test_shared_kernel_keeps_each_layers_prepared_parameters(self):
        self.config.enable_oproj_triton_quant = True
        self.config.enable_indexer_q_triton_quant = True
        kernel = MagicMock()
        resources = MagicMock(
            return_value=SimpleNamespace(vector_cores=48, ub_bytes=192 * 1024, npu_arch=2201, cache_line_bytes=512)
        )
        self.namespace["torch"].empty = MagicMock(return_value=SimpleNamespace(dtype="int8"))
        self.triton.side_effect = quant_tests.load_quantizer(
            {
                "torch": self.namespace["torch"],
                "torch_npu": SimpleNamespace(get_npu_format=lambda x: x.format),
                "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
                "_static_quantize_kernel": kernel,
                "get_device_resources": resources,
            }
        )
        oproj = self.prepare(self.layer("model.layers.0.self_attn.o_proj", weight_shape=(4096, 8192)))
        indexer = self.prepare(self.layer())
        self.assertIs(oproj._static_triton_quant, indexer._static_triton_quant)
        self.assertIsNot(oproj._static_triton_reciprocal, indexer._static_triton_reciprocal)
        self.assertIsNot(oproj._static_triton_offset, indexer._static_triton_offset)
        for layer, width in ((oproj, 8192), (indexer, 2048), (oproj, 8192)):
            kernel.reset_mock()
            resources.reset_mock()
            x = self.input(width=width)
            self.method.apply(layer, x)
            self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
            resources.assert_called_once_with(x.device.index)
            arguments = kernel.__getitem__.return_value.call_args.kwargs
            self.assertIs(kernel.__getitem__.return_value.call_args.args[1], layer.aclnn_input_scale_reciprocal)
            self.assertIs(kernel.__getitem__.return_value.call_args.args[2], layer.aclnn_input_offset)
            self.assertEqual(arguments["SHAPE"], x.shape)
            self.assertEqual(arguments["STRIDES"], x.stride())
            self.assertEqual((arguments["ROW_COUNT"], arguments["COLUMN_COUNT"]), x.shape)
            self.assertFalse(arguments["PER_TENSOR"])
        self.cann.assert_not_called()

    def test_other_shapes_single_token_and_strides_use_enabled_quantizer(self):
        self.config.enable_indexer_q_triton_quant = True
        layer = self.prepare()
        cases = [self.input(tokens=n) for n in (0, 1, 5, 7, 12)]
        cases += [
            self.input(width=8192),
            self.input(dtype="float16"),
            self.input(dtype="float32"),
            self.input(ndim=3),
            self.input(contiguous=False),
        ]
        for x in cases:
            with self.subTest(shape=x.shape, dtype=x.dtype, ndim=x.ndim):
                self.method.apply(layer, x)
                self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
        self.cann.assert_not_called()

    def test_int8_skips_quantization_and_original_matmul_is_unchanged(self):
        self.config.enable_indexer_q_triton_quant = True
        layer = self.prepare()
        for x, rank, bias in (
            (self.input(dtype="int8"), 0, layer.quant_bias),
            (self.input(), 0, layer.quant_bias),
            (self.input(), 1, None),
        ):
            self.cann.reset_mock()
            self.triton.reset_mock()
            result = self.method.apply(layer, x, tp_rank=rank)
            expected = x if x.dtype == "int8" else self.triton.return_value
            self.cann.assert_not_called()
            self.assertEqual(self.triton.call_count, int(x.dtype != "int8"))
            self.matmul.assert_called_with(
                expected, layer.weight, layer.deq_scale, bias=bias, output_dtype=layer.params_dtype
            )
            self.assertIs(result, self.matmul.return_value)

    def test_tuple_indexer_input_keeps_existing_direct_quant_matmul(self):
        path = ROOT / "vllm_ascend/attention/sfa_v1.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_select_post_process")
        branch = next(
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Name)
            and n.test.func.id == "isinstance"
        )
        check = ast.FunctionDef(
            name="run",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=n) for n in ("self", "kw", "q_c")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[branch, ast.Return(value=ast.Name(id="q_li", ctx=ast.Load()))],
            decorator_list=[],
        )
        direct = MagicMock(return_value=object())
        namespace = {
            "torch": SimpleNamespace(float8_e4m3fn="float8"),
            "torch_npu": SimpleNamespace(npu_quant_matmul=direct),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[check], type_ignores=[])), str(path), "exec"), namespace
        )
        weight, scale = object(), object()
        linear = MagicMock(weight=weight, weight_scale=scale)
        tensor = MagicMock()
        tensor.shape = (6, 2048)
        tensor.view.return_value.dtype = "int8"
        pertoken_scale = MagicMock()
        pertoken_scale.dim.return_value = 1
        kw = SimpleNamespace(dtype="bfloat16")
        result = namespace["run"](SimpleNamespace(wq_b=linear), kw, (tensor, pertoken_scale))
        self.assertIs(result, direct.return_value)
        linear.assert_not_called()
        direct.assert_called_once_with(
            tensor.view.return_value, weight, scale, pertoken_scale=pertoken_scale, bias=None, output_dtype=kw.dtype
        )
        self.cann.assert_not_called()
        self.triton.assert_not_called()

    def test_indexer_flag_default_off_strict_bool_and_config_independence(self):
        path = ROOT / "vllm_ascend/ascend_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        flags = ("enable_oproj_triton_quant", "enable_vup_triton_bmm", "enable_indexer_q_triton_quant")
        body = []
        for flag in flags:
            i = next(
                i
                for i, n in enumerate(init.body)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == flag for t in n.targets)
            )
            body.extend(init.body[i : i + 2])
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec")
        for enabled in (None, *flags):
            owner = SimpleNamespace()
            cfg = {} if enabled is None else {enabled: True}
            exec(code, {"self": owner, "additional_config": cfg})
            for flag in flags:
                self.assertIs(getattr(owner, flag), flag == enabled)
        for value in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                exec(code, {"self": SimpleNamespace(), "additional_config": {"enable_indexer_q_triton_quant": value}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    quant_tests.ROOT = ROOT
    quant_tests.METHOD = ROOT / "vllm_ascend/quantization/methods/w8a8_static.py"
    quant_tests.CONFIG = ROOT / "vllm_ascend/ascend_config.py"
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
