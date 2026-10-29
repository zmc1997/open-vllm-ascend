# SPDX-License-Identifier: Apache-2.0
"""CPU launcher and actual-kernel host mocks; these do not prove NPU equality."""

import argparse
import ast
import math
import os
import sys
import unittest
from collections import namedtuple
from ctypes import c_float
from functools import cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/vup_bmm.py"
Resources = namedtuple(
    "Resources",
    "vector_cores cube_cores ub_bytes l1_bytes l0a_bytes l0b_bytes l0c_bytes l2_bytes cache_line_bytes",
    defaults=(512,),
)


class Tensor:
    def __init__(self, shape, dtype="bf16", strides=None, device="npu", index=0, fmt=2):
        self.shape, self.dtype, self.ndim = tuple(shape), dtype, len(shape)
        self.device, self.fmt = SimpleNamespace(type=device, index=index), fmt
        if strides is None:
            strides, size = [], 1
            for dimension in reversed(shape):
                strides.insert(0, size)
                size *= max(dimension, 1)
        self.strides = tuple(strides)

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype in ("f32", "i32") else 2

    def view(self, dtype):
        result = Tensor(self.shape, dtype, self.strides, self.device.type, self.device.index, self.fmt)
        if self.element_size() != result.element_size():
            raise AssertionError("Bit view must retain element width")
        result.alias_of = self
        return result

    def is_contiguous(self):
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size > 1 and stride != expected:
                return False
            expected *= size
        return True


class Array(np.ndarray):
    def to(self, dtype, bitcast=False, **_):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(Array)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset
        self.dtype = SimpleNamespace(element_ty=data.dtype.type)

    def __add__(self, offset):
        return Address(self, np.asarray(offset, dtype=np.int64) + self.offset)


class Address:
    def __init__(self, pointer, offsets):
        self.pointer, self.offsets = pointer, offsets

    def __add__(self, offsets):
        return Address(self.pointer, self.offsets + offsets)

    def __iter__(self):
        return iter((self.pointer, self.offsets))


class HostTL:
    constexpr, int64, float16, float32 = object, np.int64, np.float16, np.float32

    def __init__(self):
        self.pid = (0, 0)

    def program_id(self, axis):
        return np.asarray(self.pid[axis]).view(Array)

    @staticmethod
    def arange(start, end):
        return np.arange(start, end).view(Array)

    minimum, where = staticmethod(np.minimum), staticmethod(np.where)
    trans = staticmethod(lambda x: x.T.view(Array))
    gather = staticmethod(lambda x, indices, axis: np.take_along_axis(x, indices, axis=axis).view(Array))

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype=dtype).view(Array)

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def load(address, mask=True, other=0):
        pointer, offsets = address
        valid = np.broadcast_to(mask, offsets.shape)
        result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
        result[valid] = pointer.data[offsets[valid]]
        return result.view(Array)

    @staticmethod
    def store(address, values, mask=True):
        pointer, offsets = address
        valid = np.broadcast_to(mask, offsets.shape)
        pointer.data[offsets[valid]] = np.asarray(values)[valid]

    @staticmethod
    def dot(x, weight, accumulator=None, **_):
        result = x.astype(np.float32) @ weight.astype(np.float32)
        return (result if accumulator is None else accumulator + result).view(Array)


class VupBmmTests(unittest.TestCase):
    def setUp(self):
        self.resources = Resources(48, 24, 196608, 524288, 65536, 65536, 131072, 201326592)
        self.kernel = MagicMock()
        self.fp32_kernel = MagicMock()
        self.copy_kernel = MagicMock()
        self.transpose_copy_kernel = MagicMock()
        self.control_sha = MagicMock(return_value="control-sha")
        self.empty = MagicMock(side_effect=lambda shape, **kw: Tensor(shape, kw["dtype"]))
        self.zeros = MagicMock(side_effect=lambda shape, **kw: Tensor(shape, kw["dtype"]))
        self.cast = MagicMock(side_effect=lambda x, fmt: Tensor(x.shape, x.dtype, fmt=fmt))
        functions = [
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "_cube_math_type",
                "_fp32_tiling",
                "_half_tiling",
                "_copy_block",
                "_copy_tiling",
                "_contiguous_copy",
                "vup_bmm",
            )
        ]
        namespace = {
            "torch": SimpleNamespace(
                Tensor=Tensor,
                bfloat16="bf16",
                float16="f16",
                float32="f32",
                int16="i16",
                int32="i32",
                empty=self.empty,
                zeros=self.zeros,
                are_deterministic_algorithms_enabled=lambda: False,
            ),
            "torch_npu": SimpleNamespace(
                get_npu_format=lambda x: x.fmt,
                npu_format_cast=self.cast,
                _C=SimpleNamespace(
                    _npu_get_deterministic_level=lambda: 0,
                    _npu_getOption=lambda key: None,
                    _npu_get_soc_version=lambda: 220,
                ),
                npu=SimpleNamespace(matmul=SimpleNamespace(cube_math_type=None, allow_hf32=False)),
            ),
            "triton": SimpleNamespace(cdiv=HostTL.cdiv, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            "_vup_bmm_kernel": self.kernel,
            "_vup_fp32_bmm_kernel": self.fp32_kernel,
            "_vup_contiguous_copy_kernel": self.copy_kernel,
            "_vup_transpose_copy_kernel": self.transpose_copy_kernel,
            "_fp32_hardware": lambda index: (24, 131072, 201326592),
            "get_device_resources": lambda index: self.resources,
            "DeviceResources": Resources,
            "DMA_ALIGNMENT_BYTES": 32,
            "cache": cache,
            "_hf32_control_sha": self.control_sha,
            "math": math,
            "c_float": c_float,
            "os": os,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.candidate = namespace["vup_bmm"]
        self.tiling = namespace["_fp32_tiling"]
        self.math_mode = namespace["_cube_math_type"]
        self.namespace = namespace

    def test_dimensions_dtypes_and_real_strides_reach_kernel(self):
        cases = [
            ((6, 32, 512), (32, 512, 256), "bf16", None, None, (32, 1)),
            ((1, 32, 512), (32, 512, 256), "bf16", None, None, (32, 1)),
            ((6, 32, 512), (32, 512, 256), "f32", None, None, (64, 1)),
            ((17, 3, 512), (3, 512, 513), "f32", None, None, (15, 2)),
            ((17, 3, 512), (3, 512, 513), "bf16", None, None, (9, 2)),
            ((17, 3, 512), (3, 512, 513), "f16", None, None, (9, 2)),
            ((17, 3, 513), (3, 513, 259), "f16", (4000, 1000, 1), (2000, 3, 0), (6, 2)),
            ((2, 5, 33), (5, 33, 7), "f32", (0, 99, 2), (0, 1, 40), (5, 1)),
            ((3, 2, 35), (2, 35, 19), "f16", (710, 142, 2), (5538, 78, 2), (2, 1)),
        ]
        for xs, ws, dtype, xstride, wstride, grid in cases:
            x, weight = Tensor(xs, dtype, xstride), Tensor(ws, dtype, wstride)
            result = self.candidate(x, weight)
            self.assertEqual((result.shape, result.dtype), ((xs[0], xs[1], ws[2]), dtype))
            if dtype == "f32":
                plan = self.tiling(*xs, ws[2], 24, 131072, 201326592)
                kernel = self.fp32_kernel
                kernel.__getitem__.assert_called_with((plan["TOTAL"],))
            else:
                kernel = self.kernel
            args, kw = kernel.__getitem__.return_value.call_args
            if dtype != "f32":
                kernel.__getitem__.assert_called_with(
                    (xs[1] * HostTL.cdiv(ws[2], kw["BLOCK_V"]), HostTL.cdiv(xs[0], kw["BLOCK_M"]))
                )
            actual_x, actual_weight, actual_result = args
            self.assertIs(actual_result, result)
            copy_calls = (
                self.copy_kernel.__getitem__.return_value.call_args_list
                + self.transpose_copy_kernel.__getitem__.return_value.call_args_list
            )
            self.assertCountEqual(
                [call.args[0].alias_of for call in copy_calls],
                [tensor for tensor in (x, weight) if not tensor.is_contiguous()],
            )
            for source, actual in ((x, actual_x), (weight, actual_weight)):
                self.assertEqual((actual.shape, actual.dtype), (source.shape, source.dtype))
                self.assertTrue(actual.is_contiguous())
                if source.is_contiguous():
                    self.assertIs(actual, source)
                else:
                    self.assertIsNot(actual, source)
            for call in copy_calls:
                source_bits, destination_bits = call.args
                source, destination = source_bits.alias_of, destination_bits.alias_of
                bit_dtype = "i16" if source.element_size() == 2 else "i32"
                self.assertEqual((source_bits.dtype, destination_bits.dtype), (bit_dtype, bit_dtype))
                self.assertEqual(source_bits.stride(), source.stride())
                self.assertEqual(destination_bits.stride(), destination.stride())
                self.assertFalse(call.kwargs["multibuffer"])
                self.assertTrue(destination.is_contiguous())
            for call in self.copy_kernel.__getitem__.return_value.call_args_list:
                source = call.args[0].alias_of
                span = (call.kwargs["BLOCK"] - 1) * source.stride(2) + 1
                self.assertEqual(call.kwargs["SPAN"], 1 << (span - 1).bit_length())
            self.assertEqual(tuple(kw[k] for k in ("N", "H", "K", "V")), (*xs, ws[2]))
            if dtype == "f32":
                self.assertEqual(
                    {key: kw[key] for key in plan if key != "TOTAL"},
                    {key: value for key, value in plan.items() if key != "TOTAL"},
                )
                self.assertEqual(kw["BLOCK_K"], min(512, plan["K_TILE"]))
                self.assertEqual(kw["CONTROL_SHA"], "")
            else:
                self.assertEqual(tuple(kw[k] for k in ("X_N", "X_H", "X_K")), actual_x.stride())
                self.assertEqual(tuple(kw[k] for k in ("W_H", "W_K", "W_V")), actual_weight.stride())
                self.assertLessEqual(kw["BLOCK_M"] * kw["BLOCK_V"] * 4, self.resources.l0c_bytes)
                self.assertLessEqual((kw["BLOCK_M"] + kw["BLOCK_V"]) * kw["BLOCK_K"] * 2, self.resources.l1_bytes // 2)
            self.assertFalse(kw["enable_fp_fusion"])
            self.copy_kernel.reset_mock()
            self.transpose_copy_kernel.reset_mock()
        self.control_sha.assert_not_called()

    def test_half_tiling_varies_with_shape_and_respects_actual_resource_budgets(self):
        choose = self.namespace["_half_tiling"]
        observed = set()
        for resource in (
            self.resources,
            self.resources._replace(
                cube_cores=7,
                vector_cores=14,
                l0c_bytes=32768,
                l0a_bytes=32768,
                l0b_bytes=32768,
                l1_bytes=262144,
                ub_bytes=98304,
                l2_bytes=4096,
            ),
        ):
            for n, h, k, v in (
                (1, 1, 1, 1),
                (6, 32, 512, 256),
                (17, 3, 1025, 259),
                (129, 2, 513, 513),
                (33, 7, 17, 65),
            ):
                m, reduction, column = choose(n, h, k, v, 2, resource)
                observed.add((m, reduction, column))
                for tile in (m, reduction, column):
                    self.assertEqual(tile & (tile - 1), 0)
                    self.assertEqual(tile % 16, 0)
                self.assertLessEqual(m * column * 4, resource.l0c_bytes)
                self.assertLessEqual((m + column) * reduction * 2, resource.l1_bytes // 2)
                self.assertGreaterEqual(HostTL.cdiv(n, m) * m, n)
                self.assertGreaterEqual(HostTL.cdiv(v, column) * column, v)
        self.assertGreater(len({p[0] for p in observed}), 1)
        self.assertGreater(len({p[1] for p in observed}), 1)
        self.assertGreater(len({p[2] for p in observed}), 1)
        # The top-level dot stages in L1. L0 subdivision belongs to lowering;
        # reducing L0 must not force repeated GM staging at Python tile level.
        large = choose(6, 32, 512, 256, 2, self.resources)
        self.assertEqual(large, (16, 256, 256))
        self.assertEqual(large, choose(6, 32, 512, 256, 2, self.resources._replace(l0a_bytes=32768, l0b_bytes=32768)))
        small = choose(6, 32, 512, 256, 2, self.resources._replace(l1_bytes=262144))
        self.assertLess(small[1], large[1])

    def test_dtype_dispatch_keeps_fp32_out_of_half_kernel(self):
        for dtype in ("bf16", "f16", "f32"):
            with self.subTest(dtype=dtype):
                self.kernel.reset_mock()
                self.fp32_kernel.reset_mock()
                selected, unused = (
                    (self.fp32_kernel, self.kernel) if dtype == "f32" else (self.kernel, self.fp32_kernel)
                )
                unused.__getitem__.side_effect = AssertionError("Wrong dtype kernel was selected")
                try:
                    result = self.candidate(Tensor((3, 2, 33), dtype), Tensor((2, 33, 19), dtype))
                finally:
                    unused.__getitem__.side_effect = None
                self.assertEqual((result.shape, result.dtype), ((3, 2, 19), dtype))
                selected.__getitem__.return_value.assert_called_once()
                unused.__getitem__.assert_not_called()

    def test_copy_partition_balances_cores_and_limits_strided_working_set(self):
        choose = self.namespace["_copy_block"]
        for size in (1, 129, 4097, 10_000_003):
            for width in (2, 4):
                for stride in (0, 1, 2, 3, 17):
                    for resource in (self.resources, self.resources._replace(vector_cores=8, ub_bytes=65536)):
                        block = choose(size, width, resource, stride)
                        self.assertEqual(block & (block - 1), 0)
                        span = 1 << ((block - 1) * stride).bit_length()
                        payload = min(span * width, block * 32) if stride > 1 else block * width
                        self.assertLessEqual(payload + block * (width + 2 * 8 + 1), resource.ub_bytes)
                        self.assertGreaterEqual(HostTL.cdiv(size, block) * block, size)
        self.assertLess(choose(49152, 2, self.resources), choose(49152, 2, self.resources._replace(vector_cores=8)))

    def test_copy_tiling_preserves_row_gaps_and_broadcasts_and_bounds_transpose_padding(self):
        choose = self.namespace["_copy_tiling"]
        self.assertEqual(choose((6, 32, 512), (32768, 1024, 2), 2, self.resources)[:2], (3, 98304))
        self.assertEqual(choose((3, 2, 35), (710, 142, 2), 2, self.resources)[:2], (5, 35))
        self.assertEqual(choose((5, 5, 259), (6000, 0, 0), 2, self.resources)[:2], (4, 1295))
        for shape, strides in (
            ((6, 32, 512), (1, 6, 192)),
            ((32, 512, 256), (1, 32, 16384)),
            ((3, 2, 35), (2, 1, 6)),
            ((2, 3, 4718593), (1, 2, 6)),
        ):
            for element in (2, 4):
                for resources in (self.resources, self.resources._replace(ub_bytes=8192, vector_cores=7)):
                    axis, inner, outer = choose(shape, strides, element, resources)
                    self.assertEqual(strides[axis], 1)
                    aligned_inner = HostTL.cdiv(inner, 32 // element) * (32 // element)
                    aligned_outer = HostTL.cdiv(outer, 32 // element) * (32 // element)
                    self.assertLessEqual(
                        2 * aligned_inner * aligned_outer * element + (inner + outer) * 17, resources.ub_bytes
                    )
                    self.assertGreaterEqual(HostTL.cdiv(shape[axis], inner) * inner, shape[axis])
                    self.assertGreaterEqual(HostTL.cdiv(shape[2], outer) * outer, shape[2])

    def test_hf32_control_contents_enter_key_once_per_process(self):
        function = next(
            node
            for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_hf32_control_sha"
        )
        path = MagicMock()
        path.return_value.read_bytes.return_value = b"first bitcode"
        namespace = {
            "cache": cache,
            "sha256": sha256,
            "Path": path,
            "_vup_hf32_enable": SimpleNamespace(bitcode="control.bc"),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        control_sha = namespace["_hf32_control_sha"]
        self.assertEqual(control_sha(), sha256(b"first bitcode").hexdigest())
        self.assertEqual(control_sha(), sha256(b"first bitcode").hexdigest())
        path.assert_called_once_with("control.bc")
        path.return_value.read_bytes.assert_called_once_with()
        # A deployment restarts the process; its first read must change the key.
        control_sha.cache_clear()
        path.return_value.read_bytes.return_value = b"replacement bitcode"
        self.assertEqual(control_sha(), sha256(b"replacement bitcode").hexdigest())

    def test_fp32_tiling_and_native_shift_controls_are_not_shape_constants(self):
        legacy = self.tiling(6, 32, 512, 256, 24, 131072, 201326592)
        self.assertEqual(
            [legacy[key] for key in ("BLOCK_M", "BLOCK_V", "K_TILE", "N_LOOP", "CORE_COUNT", "TOTAL")],
            [16, 128, 256, 2, 24, 64],
        )
        separate = self.tiling(6, 32, 512, 128, 24, 131072, 201326592)
        self.assertEqual((separate["N_LOOP"], separate["TOTAL"]), (1, 32))
        for cores in (2, 7, 24, 64):
            for n, h, k, v in ((1, 1, 1, 1), (17, 5, 777, 513), (129, 2, 19, 17)):
                plan = self.tiling(n, h, k, v, cores, 131072, 201326592)
                self.assertEqual(plan["CORE_COUNT"], min(plan["TOTAL"], cores))
                self.assertLessEqual(plan["BLOCK_M"] * plan["BLOCK_V"] * 4, 131072)
                self.assertLessEqual((plan["BLOCK_M"] + plan["BLOCK_V"]) * min(512, plan["K_TILE"]) * 4, 524288)
        x, weight = Tensor((6, 32, 512), "f32"), Tensor((32, 512, 256), "f32")
        for environment, deterministic, level, expected in (
            ({}, False, 0, True),
            ({"CLOSE_MATMUL_K_SHIFT": "1"}, False, 0, False),
            ({}, True, 1, True),
            ({}, True, 2, False),
        ):
            self.namespace["torch"].are_deterministic_algorithms_enabled = (
                lambda deterministic=deterministic: deterministic
            )
            self.namespace["torch_npu"]._C._npu_get_deterministic_level = lambda level=level: level
            with patch.dict(os.environ, environment, clear=True):
                self.candidate(x, weight)
            self.assertEqual(self.fp32_kernel.__getitem__.return_value.call_args.kwargs["SHUFFLE_K"], expected)

    def test_math_modes_are_pure_reads_and_preserve_native_override_priority(self):
        state = dict(override=None, hf32=False, precision=None, soc=220)

        class ReadOnlyMatmul:
            @property
            def cube_math_type(self):
                return None if state["override"] is None else SimpleNamespace(value=state["override"])

            @property
            def allow_hf32(self):
                return state["hf32"]

            def __setattr__(self, key, value):
                raise AssertionError("No global matmul setting may be changed")

        forbidden = MagicMock(side_effect=AssertionError("No global setter may be called"))
        options = MagicMock(side_effect=lambda key: state["precision"] if key == "ACL_PRECISION_MODE" else None)
        self.namespace["torch_npu"].npu = SimpleNamespace(matmul=ReadOnlyMatmul(), _get_deterministic_level=forbidden)
        self.namespace["torch_npu"]._C._npu_getOption = options
        self.namespace["torch_npu"]._C._npu_get_soc_version = lambda: state["soc"]
        self.namespace["torch_npu"]._C._npu_setOption = forbidden
        self.namespace["torch_npu"]._C._npu_set_deterministic_level = forbidden
        x, weight = Tensor((6, 32, 512), "f32"), Tensor((32, 512, 256), "f32")
        for mode in (0, 1, 2, 3, 4):
            state.update(override=mode, hf32=True, precision=b"allow_fp32_to_fp16")
            before = state.copy()
            options.reset_mock()
            self.control_sha.reset_mock()
            with patch.dict(os.environ, {}, clear=True):
                self.candidate(x, weight)
            actual = self.fp32_kernel.__getitem__.return_value.call_args.kwargs
            self.assertEqual(actual["MATH_MODE"], mode)
            self.assertEqual(actual["SHUFFLE_K"], mode != 3)
            self.assertEqual(actual["CONTROL_SHA"], "control-sha" if mode == 3 else "")
            if mode == 3:
                self.control_sha.assert_called_once_with()
            else:
                self.control_sha.assert_not_called()
            self.assertEqual(state, before)
            options.assert_not_called()
        state["override"] = None
        for soc in (219, 220, 245):
            for precision in (None, b"", b"must_keep_origin_dtype", "allow_fp32_to_fp16", b"unknown"):
                for hf32 in (False, True):
                    state.update(soc=soc, precision=precision, hf32=hf32)
                    before = state.copy()
                    options.reset_mock()
                    down = soc < 220
                    if precision == b"must_keep_origin_dtype":
                        down = False
                    elif precision == "allow_fp32_to_fp16":
                        down = True
                    self.assertEqual(
                        self.math_mode(),
                        {(False, False): 0, (False, True): 2, (True, False): 3, (True, True): 1}[(hf32, down)],
                    )
                    self.assertEqual(state, before)
                    options.assert_called_once_with("ACL_PRECISION_MODE")
        state.update(override=0)
        self.namespace["torch"].are_deterministic_algorithms_enabled = lambda: True
        self.namespace["torch_npu"]._C._npu_get_deterministic_level = lambda: 2
        with patch.dict(os.environ, {}, clear=True):
            self.candidate(x, weight)
        self.assertFalse(self.fp32_kernel.__getitem__.return_value.call_args.kwargs["SHUFFLE_K"])
        forbidden.assert_not_called()

    def test_actual_fp32_kernel_native_core_mapping_and_k_tail_order(self):
        function = next(
            node
            for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_vup_fp32_bmm_kernel"
        )
        function.decorator_list = []
        tl = HostTL()
        namespace = {"tl": tl}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        for n, h, k, v, cores in (
            (6, 4, 512, 256, 24),
            (6, 4, 512, 128, 24),
            (17, 3, 777, 259, 7),
            (129, 2, 19, 17, 3),
            (1, 2, 515, 1, 24),
        ):
            plan = self.tiling(n, h, k, v, cores, 131072, 201326592)
            x = np.broadcast_to(np.arange(1, k + 1, dtype=np.float32), (n, h, k)).copy()
            weight = np.ones((h, k, v), dtype=np.float32)
            result = np.full(n * h * v + 17, -999, dtype=np.float32)
            for tile in range(plan["TOTAL"]):
                tl.pid = tile, 0
                seen = []

                def dot(left, right, accumulator, seen=seen, **kwargs):
                    seen.extend(left[0, left[0] != 0].astype(int).tolist())
                    return HostTL.dot(left, right, accumulator, **kwargs)

                tl.dot = dot
                namespace["_vup_fp32_bmm_kernel"](
                    Pointer(x.reshape(-1)),
                    Pointer(weight.reshape(-1)),
                    Pointer(result),
                    N=n,
                    H=h,
                    K=k,
                    V=v,
                    **{key: value for key, value in plan.items() if key != "TOTAL"},
                    BLOCK_K=min(512, plan["K_TILE"]),
                    SHUFFLE_K=True,
                )
                start = (tile % plan["CORE_COUNT"]) % HostTL.cdiv(k, plan["K_TILE"]) * plan["K_TILE"]
                self.assertEqual(seen, list(range(start + 1, k + 1)) + list(range(1, start + 1)))
            np.testing.assert_array_equal(result[: n * h * v], k * (k + 1) // 2)
            np.testing.assert_array_equal(result[n * h * v :], -999)

    def test_empty_dimensions_match_native_rejection_without_launch(self):
        for dtype in ("bf16", "f16", "f32"):
            for n, h, k, v in ((0, 3, 17, 7), (2, 0, 17, 7), (2, 3, 17, 0), (2, 3, 0, 7)):
                with (
                    self.subTest(dtype=dtype, shape=(n, h, k, v)),
                    self.assertRaisesRegex(RuntimeError, "empty tensors"),
                ):
                    self.candidate(Tensor((n, h, k), dtype), Tensor((h, k, v), dtype))
        self.kernel.__getitem__.assert_not_called()
        self.empty.assert_not_called()
        self.zeros.assert_not_called()

    def test_non_nd_is_rejected_without_conversion_or_launch(self):
        for x_format, weight_format in ((29, 2), (2, 29), (0, 2)):
            with self.subTest(x_format=x_format, weight_format=weight_format), self.assertRaisesRegex(ValueError, "ND"):
                self.candidate(Tensor((3, 2, 17), fmt=x_format), Tensor((2, 17, 19), fmt=weight_format))
        self.cast.assert_not_called()
        self.empty.assert_not_called()
        self.kernel.__getitem__.assert_not_called()

    def test_invalid_contract_rejected_before_allocation(self):
        valid = Tensor((2, 3, 17)), Tensor((3, 17, 7))
        invalid = [
            (None, valid[1]),
            (Tensor((2, 3)), valid[1]),
            (valid[0], Tensor((4, 17, 7))),
            (valid[0], Tensor((3, 18, 7))),
            (Tensor((2, 3, 17), "f64"), Tensor((3, 17, 7), "f64")),
            (valid[0], Tensor((3, 17, 7), "f16")),
            (Tensor((2, 3, 17), device="cpu"), valid[1]),
            (valid[0], Tensor((3, 17, 7), index=1)),
            (Tensor((2, 3, 17), strides=(51, 17, -1)), valid[1]),
        ]
        for x, weight in invalid:
            with self.subTest(x=x, weight=weight), self.assertRaises(ValueError):
                self.candidate(x, weight)
        self.empty.assert_not_called()
        self.zeros.assert_not_called()
        self.kernel.__getitem__.assert_not_called()

    def test_actual_kernel_host_mock_covers_tail_k_loop_and_broadcast_strides(self):
        kernel = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_vup_bmm_kernel"
        )
        kernel.decorator_list = []
        tl = HostTL()
        namespace = {"tl": tl}
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(SOURCE), "exec"), namespace)
        n, h, k, v = 3, 2, 515, 19
        ordinary = ((h * k, k, 1), (k * v, v, 1))
        layouts = [ordinary, ((h * (k * 2 + 3) + 7, k * 2 + 3, 2), (k * (v * 2 + 5) + 11, v * 2 + 5, 2))]
        for operand in range(2):
            for axis in range(3):
                strides = [list(s) for s in ordinary]
                strides[operand][axis] = 0
                layouts.append(tuple(tuple(s) for s in strides))
        dtype = np.float16
        for xs, ws in layouts:
            memories, views = [], []
            for shape, strides in (((n, h, k), xs), ((h, k, v), ws)):
                size = 4 + sum((dimension - 1) * stride for dimension, stride in zip(shape, strides))
                data = ((np.arange(size) % 3) - 1).astype(dtype)
                view = np.lib.stride_tricks.as_strided(
                    data[3:], shape=shape, strides=tuple(s * data.itemsize for s in strides)
                )
                memories.append(Pointer(data, 3))
                views.append(view)
            result = np.full(n * h * v, np.nan, dtype=dtype)
            for head in range(h):
                tl.pid = head, 0
                namespace["_vup_bmm_kernel"](
                    *memories,
                    Pointer(result),
                    n,
                    h,
                    k,
                    v,
                    *xs,
                    *ws,
                    16,
                    512,
                    32,
                )
            expected = np.einsum("nhk,hkv->nhv", *views, dtype=np.float32).astype(dtype)
            np.testing.assert_array_equal(result.reshape(n, h, v), expected)

    def test_actual_copy_kernels_preserve_special_bits_strides_and_tail_guard(self):
        functions = [
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_vup_contiguous_copy_kernel", "_vup_transpose_copy_kernel")
        ]
        for function in functions:
            function.decorator_list = []
        tl = HostTL()
        namespace = {"tl": tl}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        for shape in ((5, 5, 259), (3, 2, 17), (1, 2, 7)):
            size = math.prod(shape)
            dense = (shape[1] * shape[2], shape[2], 1)
            for bits_dtype, integer_dtype, patterns, sentinel in (
                (np.uint16, np.int16, (0, 0x8000, 0x7C00, 0x7F80, 0xFF80, 0x7FC1, 0x7E01, 0xFFFF), 0xA55A),
                (np.uint32, np.int32, (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFFFFFFF), 0xA55AA55A),
            ):
                for strides in (
                    (6000, 700, 2),
                    (0, 700, 2),
                    (6000, 0, 2),
                    (6000, 700, 0),
                    tuple(x * 2 for x in dense),
                    tuple(x * 3 for x in dense),
                    (1, shape[0], shape[0] * shape[1]),
                    (shape[1], 1, shape[0] * shape[1]),
                ):
                    count = 4 + sum((n - 1) * stride for n, stride in zip(shape, strides))
                    source = np.resize(np.asarray(patterns, dtype=bits_dtype), count)
                    before = source.copy()
                    expected = (
                        np.lib.stride_tricks.as_strided(
                            source[3:], shape=shape, strides=tuple(stride * source.itemsize for stride in strides)
                        )
                        .reshape(-1)
                        .copy()
                    )
                    result = np.full(size + 17, sentinel, dtype=bits_dtype)
                    mode, first, second = self.namespace["_copy_tiling"](
                        shape, strides, source.itemsize, self.resources
                    )
                    if mode < 3:
                        axis, batch_axis = mode, 1 - mode
                        for batch in range(shape[batch_axis]):
                            for outer in range(HostTL.cdiv(shape[2], second)):
                                for inner in range(HostTL.cdiv(shape[axis], first)):
                                    tl.pid = inner, outer, batch
                                    namespace["_vup_transpose_copy_kernel"](
                                        Pointer(source.view(integer_dtype), 3),
                                        Pointer(result.view(integer_dtype)),
                                        shape[axis],
                                        shape[2],
                                        strides[batch_axis],
                                        strides[2],
                                        dense[batch_axis],
                                        dense[axis],
                                        first,
                                        second,
                                        shape[axis] % first == 0 and shape[2] % second == 0,
                                    )
                    else:
                        for row in range(size // first):
                            for program in range(HostTL.cdiv(first, second)):
                                tl.pid = program, row
                                namespace["_vup_contiguous_copy_kernel"](
                                    Pointer(source.view(integer_dtype), 3),
                                    Pointer(result.view(integer_dtype)),
                                    shape[1],
                                    *strides,
                                    mode - 3,
                                    first,
                                    second,
                                    1 << ((second - 1) * strides[2]).bit_length(),
                                    source.itemsize,
                                    32,
                                )
                    np.testing.assert_array_equal(result[:size], expected)
                    np.testing.assert_array_equal(result[size:], sentinel)
                    np.testing.assert_array_equal(source, before)

    def test_actual_kernel_adaptive_tiles_cover_m_k_v_and_keep_tail_guard(self):
        function = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_vup_bmm_kernel"
        )
        function.decorator_list = []
        tl = HostTL()
        namespace = {"tl": tl}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        for n, h, k, v in ((3, 2, 33, 513), (33, 3, 513, 129), (129, 2, 65, 33)):
            dtype = np.float16
            block_m, block_k, block_v = self.namespace["_half_tiling"](n, h, k, v, 2, self.resources)
            x = ((np.arange(n * h * k) % 3) - 1).astype(dtype).reshape(n, h, k)
            weight = ((np.arange(h * k * v) % 3) - 1).astype(dtype).reshape(h, k, v)
            result = np.full(n * h * v + 17, -999, dtype=dtype)
            for tile in range(h * HostTL.cdiv(v, block_v)):
                for row in range(HostTL.cdiv(n, block_m)):
                    tl.pid = tile, row
                    namespace["_vup_bmm_kernel"](
                        Pointer(x.reshape(-1)),
                        Pointer(weight.reshape(-1)),
                        Pointer(result),
                        n,
                        h,
                        k,
                        v,
                        h * k,
                        k,
                        1,
                        k * v,
                        v,
                        1,
                        block_m,
                        block_k,
                        block_v,
                    )
            expected = np.einsum("nhk,hkv->nhv", x, weight, dtype=np.float32).astype(dtype)
            np.testing.assert_array_equal(result[: n * h * v].reshape(n, h, v), expected)
            np.testing.assert_array_equal(result[n * h * v :], -999)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path)
    args, remaining = parser.parse_known_args()
    if args.source_root is not None:
        SOURCE = args.source_root.resolve() / "vllm_ascend/ops/triton/vup_bmm.py"
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
