# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch regressions for shared static quantization.

Run this file directly with Python. Extract the production methods to avoid
initializing the NPU plugin; mocks cover dispatch and parameter ownership only.
Numerical equivalence and graph replay require the separate NPU kernel tests.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "vllm_ascend/quantization/methods/w8a8_static.py"
CONFIG = ROOT / "vllm_ascend/ascend_config.py"
KERNEL_MODULE = "vllm_ascend.ops.triton.static_quantize"
DISPATCH_PLAN = {
    "NUM_CORES": 3,
    "ROW_FACTOR": 2,
    "COLUMN_FACTOR": 128,
    "COLUMN_CORES": 2,
    "BASE_N": 2,
    "BLOCK_ROWS": 2,
    "BLOCK_COLS": 128,
}


def load_methods(namespace):
    tree = ast.parse(METHOD.read_text(encoding="utf-8"))
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendW8A8LinearMethod"
    )
    cls = ast.ClassDef(
        name=original.name,
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in ("apply", "process_weights_after_loading")
        ],
    )
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id.startswith("_INDEXER_Q_") for target in node.targets)
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *constants, cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(METHOD), "exec"), namespace)
    return namespace[original.name]()


def load_quantizer(namespace):
    path = ROOT / "vllm_ascend/ops/triton/static_quantize.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quantize"]
    # This suite tests dispatch and metadata forwarding. Planner policy has its
    # own tests; these opaque fixture tiles do not encode any model shape.
    namespace.setdefault("_quantize_tiling", MagicMock(side_effect=lambda *args: dict(DISPATCH_PLAN)))
    namespace.setdefault("DMA_ALIGNMENT_BYTES", 32)
    namespace.setdefault("triton", SimpleNamespace()).next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["quantize"]


class TestOProjTritonQuant(unittest.TestCase):
    def setUp(self):
        self.cann = MagicMock(return_value=SimpleNamespace(dtype="int8"))
        self.triton = MagicMock(return_value=SimpleNamespace(dtype="int8"))
        self.matmul = MagicMock(return_value=object())
        self.config = SimpleNamespace(enable_oproj_triton_quant=False, enable_indexer_q_triton_quant=False)
        self.namespace = {
            "torch": SimpleNamespace(
                int8="int8",
                bfloat16="bfloat16",
                float32="float32",
                float16="float16",
                strided="strided",
                nn=SimpleNamespace(Parameter=lambda data, **kwargs: data),
                ops=SimpleNamespace(vllm=SimpleNamespace(quantize=self.cann)),
                flatten=lambda data: data,
            ),
            "torch_npu": SimpleNamespace(npu_quant_matmul=self.matmul),
            "COMPRESSED_TENSORS_METHOD": "compressed-tensors",
            "maybe_trans_nz": lambda data: data,
            "get_ascend_config": lambda: self.config,
        }
        self.method = load_methods(self.namespace)
        module = ModuleType(KERNEL_MODULE)
        module.quantize = self.triton
        self.modules = patch.dict("sys.modules", {KERNEL_MODULE: module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    @staticmethod
    def layer(prefix="model.layers.0.self_attn.o_proj", dtype="bfloat16", tp=2):
        weight = MagicMock()
        weight.shape = (4, 8)
        device = SimpleNamespace(type="npu", index=0)
        input_scale, input_offset = MagicMock(), MagicMock()

        def repeated_parameter(width):
            parameter = MagicMock()
            for value in (parameter, parameter.__rtruediv__.return_value):
                value.dtype, value.layout, value.ndim = dtype, "strided", 1
                value.shape, value.device, value.format = (width,), device, 2
                value.numel.return_value = width
                value.element_size.return_value = 4 if dtype == "float32" else 2
                value.stride.side_effect = lambda dim=None: (1,) if dim is None else (1,)[dim]
            parameter.to.return_value = parameter
            return parameter

        input_scale.repeat.side_effect = repeated_parameter
        input_offset.repeat.side_effect = repeated_parameter
        return SimpleNamespace(
            prefix=prefix,
            params_dtype=dtype,
            tp_size=tp,
            weight=SimpleNamespace(data=weight),
            input_scale=SimpleNamespace(data=input_scale),
            input_offset=SimpleNamespace(data=input_offset),
            weight_scale=SimpleNamespace(data=MagicMock()),
            weight_offset=SimpleNamespace(data=MagicMock()),
            quant_bias=object(),
            deq_scale=object(),
        )

    @staticmethod
    def input(
        dtype="bfloat16",
        ndim=2,
        contiguous=True,
        tokens=6,
        device_type="npu",
        device_index=0,
        shape=None,
        strides=None,
        layout="strided",
        format=2,
    ):
        shape = shape if shape is not None else (tokens,) + (1,) * (ndim - 2) + (8192,)
        if strides is None:
            strides = []
            size = 1
            for width in reversed(shape):
                strides.insert(0, size)
                size *= width
        return SimpleNamespace(
            dtype=dtype,
            ndim=len(shape),
            shape=shape,
            layout=layout,
            format=format,
            stride=lambda dim=None: tuple(strides) if dim is None else strides[dim],
            numel=lambda: math.prod(shape),
            element_size=lambda: {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}[dtype],
            device=SimpleNamespace(type=device_type, index=device_index),
            is_contiguous=lambda: contiguous,
        )

    def prepare(self, layer):
        self.method.process_weights_after_loading(layer)
        return layer

    def test_disabled_and_other_projections_keep_cann(self):
        for enabled, prefix in (
            (False, "model.layers.0.self_attn.o_proj"),
            (True, "model.layers.0.self_attn.v_up_proj"),
            (True, "model.layers.0.self_attn.o_proj_extra"),
        ):
            with self.subTest(enabled=enabled, prefix=prefix):
                self.config.enable_oproj_triton_quant = enabled
                layer = self.prepare(self.layer(prefix))
                x = self.input()
                self.method.apply(layer, x)
                self.assertIsNone(layer._static_triton_quant)
                self.cann.assert_called_once_with(
                    x, layer.aclnn_input_scale, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset
                )
                self.triton.assert_not_called()
                self.cann.reset_mock()

    def test_main_and_mtp_alias_original_full_parameters(self):
        self.config.enable_oproj_triton_quant = True
        for prefix in ("model.layers.0.self_attn.o_proj", "model.layers.78.self_attn.o_proj"):
            with self.subTest(prefix=prefix):
                layer = self.prepare(self.layer(prefix))
                reciprocal = layer.aclnn_input_scale_reciprocal
                offset = layer.aclnn_input_offset
                self.assertIs(layer._static_triton_reciprocal, reciprocal)
                self.assertIs(layer._static_triton_offset, offset)
                for _ in range(2):
                    x = self.input()
                    self.method.apply(layer, x)
                    self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
                reciprocal.__getitem__.assert_not_called()
                offset.__getitem__.assert_not_called()
                reciprocal.to.assert_not_called()
                # Preserve the original CANN parameter preparation cast only.
                offset.to.assert_called_once_with(layer.aclnn_input_scale.dtype)
        self.cann.assert_not_called()

    def test_tp_and_parameter_dtype_do_not_limit_quantization(self):
        self.config.enable_oproj_triton_quant = True
        for dtype in ("bfloat16", "float16", "float32"):
            for tp in (1, 2, 4, 16):
                with self.subTest(dtype=dtype, tp=tp):
                    layer = self.prepare(self.layer(dtype=dtype, tp=tp))
                    x = self.input(dtype=dtype, tokens=1)
                    self.method.apply(layer, x)
                    self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
        self.cann.assert_not_called()

    def test_single_token_mtp_uses_enabled_quantizer(self):
        layer = self.layer("model.layers.78.self_attn.o_proj")
        self.config.enable_oproj_triton_quant = True
        self.method.process_weights_after_loading(layer)
        self.method.apply(layer, self.input(tokens=1))
        self.cann.assert_not_called()
        self.triton.assert_called_once()

    def test_int8_input_skips_both_quantizers(self):
        self.config.enable_oproj_triton_quant = True
        layer = self.prepare(self.layer())
        x = self.input(dtype="int8")
        self.method.apply(layer, x)
        self.cann.assert_not_called()
        self.triton.assert_not_called()
        self.assertIs(self.matmul.call_args.args[0], x)

    def test_enabled_shapes_and_strides_do_not_fall_back(self):
        self.config.enable_oproj_triton_quant = True
        layer = self.prepare(self.layer())
        cases = [
            self.input(dtype="float16"),
            self.input(ndim=3),
            self.input(contiguous=False),
            self.input(tokens=0),
            self.input(tokens=1),
            self.input(shape=(17, 129)),
            self.input(dtype="float32", shape=(2, 3, 127)),
            self.input(shape=(128,)),
        ]
        for x in cases:
            with self.subTest(dtype=x.dtype, ndim=x.ndim, shape=x.shape, device=x.device):
                self.method.apply(layer, x)
                self.triton.assert_called_with(x, layer._static_triton_reciprocal, layer._static_triton_offset)
        self.cann.assert_not_called()

    def test_operator_metadata_errors_are_not_hidden_by_cann(self):
        self.config.enable_oproj_triton_quant = True
        layer = self.prepare(self.layer())
        self.triton.side_effect = load_quantizer(
            {
                "torch": self.namespace["torch"],
                "torch_npu": SimpleNamespace(get_npu_format=lambda x: x.format),
            }
        )
        for x in (
            self.input(dtype="int32"),
            self.input(layout="sparse_coo"),
            self.input(device_type="cpu"),
            self.input(device_index=1),
            self.input(shape=()),
            self.input(shape=(1,) * 9),
            self.input(format=29),
        ):
            with (
                self.subTest(dtype=x.dtype, layout=x.layout, device=x.device),
                self.assertRaises(ValueError),
            ):
                self.method.apply(layer, x)
        self.cann.assert_not_called()

    def test_real_shape_stride_and_empty_launch_contract(self):
        kernel, empty = MagicMock(), MagicMock()
        devices = {
            0: SimpleNamespace(vector_cores=7, ub_bytes=64 * 1024, npu_arch=2201, cache_line_bytes=512),
            3: SimpleNamespace(vector_cores=24, ub_bytes=192 * 1024, npu_arch=2201, cache_line_bytes=512),
        }
        resources = MagicMock(side_effect=lambda device: devices[device])
        torch = self.namespace["torch"]
        torch.empty = empty
        namespace = {
            "torch": torch,
            "torch_npu": SimpleNamespace(get_npu_format=lambda x: x.format),
            "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
            "_static_quantize_kernel": kernel,
            "get_device_resources": resources,
        }
        quantize = load_quantizer(namespace)
        planner = namespace["_quantize_tiling"]
        inputs = (
            (self.input(dtype="float16", shape=(1, 3)), (1,), "float16", 1),
            (self.input(shape=(0, 7)), (7,), "bfloat16", 1),
            (
                self.input(dtype="float32", shape=(2, 3, 7), strides=(50, 14, 2), contiguous=False, device_index=3),
                (7,),
                "float32",
                2,
            ),
            (self.input(shape=(17, 129), strides=(0, 0), contiguous=False), (129,), "bfloat16", 0),
            (self.input(shape=(1, 2, 1, 1, 1, 1, 3, 7)), (), "float32", 0),
        )
        for x, parameter_shape, parameter_dtype, parameter_stride in inputs:
            # Reset before each case so a failed subtest cannot contaminate the
            # empty-input or subsequent device checks.
            for mock in (kernel, empty, resources, planner):
                mock.reset_mock()
            reciprocal, offset = (
                self.input(
                    dtype=parameter_dtype,
                    shape=parameter_shape,
                    strides=(parameter_stride,) if parameter_shape else (),
                    contiguous=parameter_stride == 1 or not parameter_shape,
                    device_index=x.device.index,
                )
                for _ in range(2)
            )
            with self.subTest(shape=x.shape, strides=x.stride(), format=x.format):
                result = quantize(x, reciprocal, offset)
                self.assertIs(result, empty.return_value)
                empty.assert_called_once_with(x.shape, device=x.device, dtype="int8")
                if x.numel():
                    device = devices[x.device.index]
                    per_tensor = reciprocal.numel() == 1
                    rows, width = (1, x.numel()) if per_tensor else (math.prod(x.shape[:-1]), x.shape[-1])
                    resources.assert_called_once_with(x.device.index)
                    planner.assert_called_once_with(
                        rows,
                        width,
                        x.element_size(),
                        reciprocal.element_size(),
                        x.is_contiguous(),
                        parameter_stride == 1,
                        per_tensor,
                        device.vector_cores,
                        device.ub_bytes,
                        device.cache_line_bytes,
                    )
                    plan = DISPATCH_PLAN
                    kernel.__getitem__.assert_called_once_with((plan["NUM_CORES"],))
                    kernel.__getitem__.return_value.assert_called_once_with(
                        x,
                        reciprocal,
                        offset,
                        result,
                        ROW_COUNT=rows,
                        COLUMN_COUNT=width,
                        PER_TENSOR=per_tensor,
                        RECIPROCAL_STRIDE=parameter_stride,
                        OFFSET_STRIDE=parameter_stride,
                        INPUT_SPAN=1 << ((plan["BLOCK_COLS"] - 1) * x.stride(-1)).bit_length(),
                        RECIPROCAL_SPAN=1 << ((plan["BLOCK_COLS"] - 1) * parameter_stride).bit_length(),
                        OFFSET_SPAN=1 << ((plan["BLOCK_COLS"] - 1) * parameter_stride).bit_length(),
                        PARAMETER_BLOCK=namespace["DMA_ALIGNMENT_BYTES"] // reciprocal.element_size(),
                        DMA_BYTES=namespace["DMA_ALIGNMENT_BYTES"],
                        INPUT_BYTES=x.element_size(),
                        PARAMETER_BYTES=reciprocal.element_size(),
                        SHAPE=x.shape,
                        STRIDES=x.stride(),
                        CONTIGUOUS=x.is_contiguous(),
                        **{key: value for key, value in plan.items() if key != "NUM_CORES"},
                        enable_fp_fusion=False,
                        multibuffer=False,
                        enable_ubuf_saving=True,
                    )
                else:
                    kernel.__getitem__.assert_not_called()
                    resources.assert_not_called()
                    planner.assert_not_called()

    def test_matmul_arguments_and_rank_bias_are_preserved(self):
        self.config.enable_oproj_triton_quant = True
        layer = self.prepare(self.layer())
        for rank, expected_bias in ((0, layer.quant_bias), (1, None)):
            with self.subTest(rank=rank):
                result = self.method.apply(layer, self.input(), tp_rank=rank)
                self.assertIs(result, self.matmul.return_value)
                self.matmul.assert_called_with(
                    self.triton.return_value,
                    layer.weight,
                    layer.deq_scale,
                    bias=expected_bias,
                    output_dtype=layer.params_dtype,
                )

    def test_compressed_tensor_bias_is_preserved(self):
        self.config.enable_oproj_triton_quant = True
        layer = self.layer()
        layer.ascend_quant_method = "compressed-tensors"
        self.prepare(layer)
        bias = object()
        self.method.apply(layer, self.input(), bias=bias, tp_rank=1)
        self.assertIs(self.matmul.call_args.kwargs["bias"], bias)

    def test_config_defaults_off_and_rejects_non_boolean(self):
        tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        index = next(
            i
            for i, node in enumerate(init.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "enable_oproj_triton_quant"
                for target in node.targets
            )
        )
        code = compile(ast.Module(body=init.body[index : index + 2], type_ignores=[]), str(CONFIG), "exec")
        for value, expected in (
            ({}, False),
            ({"enable_oproj_triton_quant": False}, False),
            ({"enable_oproj_triton_quant": True}, True),
        ):
            config = SimpleNamespace()
            exec(code, {"self": config, "additional_config": value})
            self.assertIs(config.enable_oproj_triton_quant, expected)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                exec(code, {"self": SimpleNamespace(), "additional_config": {"enable_oproj_triton_quant": value}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, remaining = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    METHOD = ROOT / "vllm_ascend/quantization/methods/w8a8_static.py"
    CONFIG = ROOT / "vllm_ascend/ascend_config.py"
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
