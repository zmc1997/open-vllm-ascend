# SPDX-License-Identifier: Apache-2.0
"""CPU-only scatter contract, launch and alias checks; NPU oracle is separate."""

import argparse
import ast
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]


class Tensor:
    def __init__(self, shape, dtype="i8", element_bytes=1, strides=None, format=2, device="npu", device_index=0):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.device = len(shape), SimpleNamespace(type=device, index=device_index)
        self._element_bytes = element_bytes
        self._strides = strides or ((shape[1], 1) if len(shape) == 2 else (1,))
        self.format = format

    def stride(self, axis):
        return self._strides[axis]

    def element_size(self):
        return self._element_bytes


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.kernel = MagicMock()
        self.copy_kernel = MagicMock()
        self.transpose_kernel = MagicMock()
        self.empty = MagicMock(
            side_effect=lambda shape, dtype, device: Tensor(
                shape,
                dtype,
                {"i8": 1, "bool": 1, "float8": 1, "f16": 2, "bf16": 2, "f32": 4, "i64": 8, "f64": 8}[dtype],
                device=device.type,
                device_index=device.index,
            )
        )
        self.format_cast = MagicMock(
            side_effect=lambda tensor, fmt: Tensor(
                tensor.shape,
                tensor.dtype,
                tensor.element_size(),
                format=fmt,
                device=tensor.device.type,
                device_index=tensor.device.index,
            )
        )
        self.get_format = MagicMock(side_effect=lambda tensor: tensor.format)
        source = ROOT / "vllm_ascend/ops/triton/indexer_cache.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in ("_scatter_tiling", "_transpose_tiling", "_scatter_cache", "scatter_k_cache", "scatter_scale_cache")
        ]
        namespace = {
            "torch": SimpleNamespace(Tensor=Tensor, int32="i32", int64="i64", empty=self.empty),
            "triton": SimpleNamespace(
                cdiv=lambda x, y: (x + y - 1) // y, next_power_of_2=lambda n: 1 << (n - 1).bit_length()
            ),
            "_indexer_cache_copy_kernel": self.copy_kernel,
            "_indexer_cache_transpose_kernel": self.transpose_kernel,
            "torch_npu": SimpleNamespace(get_npu_format=self.get_format, npu_format_cast=self.format_cast),
            "_indexer_cache_scatter_kernel": self.kernel,
            "lru_cache": lru_cache,
            "DMA_ALIGNMENT_BYTES": 32,
            "get_device_resources": MagicMock(return_value=SimpleNamespace(vector_cores=48, ub_bytes=196608)),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        self.write_k, self.write_scale = namespace["scatter_k_cache"], namespace["scatter_scale_cache"]
        self.tiling = namespace["_scatter_tiling"]
        self.transpose_tiling = namespace["_transpose_tiling"]
        self.resources = namespace["get_device_resources"]

    def test_dynamic_rows_widths_strides_and_bitwidths_preserve_alias(self):
        for function in (self.write_k, self.write_scale):
            for rows, width in ((1, 1), (5, 3), (6, 128), (7, 31), (17, 257), (129, 513)):
                for dtype, element_bytes in (
                    ("i8", 1),
                    ("bool", 1),
                    ("float8", 1),
                    ("f16", 2),
                    ("bf16", 2),
                    ("f32", 4),
                    ("i64", 8),
                    ("f64", 8),
                ):
                    for index_dtype in ("i32", "i64"):
                        with self.subTest(rows=rows, width=width, dtype=dtype, index_dtype=index_dtype):
                            capacity = rows + 9
                            cache = Tensor((capacity, width), dtype, element_bytes, (width * 3 + 5, 2))
                            slots = Tensor((rows, 1), index_dtype, 4 if index_dtype == "i32" else 8, strides=(3, 1))
                            values = Tensor((rows, width), dtype, element_bytes, (width * 4 + 7, 3))
                            self.assertIs(function(cache, slots, values), cache)
                            actual_values = self.empty.return_value
                            if width > 1:
                                self.empty.assert_called_once_with(values.shape, dtype=dtype, device=values.device)
                                actual_values = self.copy_kernel.__getitem__.return_value.call_args.args[1]
                                _, _, copy_tile = self.tiling(
                                    rows, width, capacity, element_bytes, slots.element_size(), 3, 48, 196608
                                )
                                self.copy_kernel.__getitem__.assert_called_once_with(
                                    (rows, (width + copy_tile - 1) // copy_tile)
                                )
                                self.copy_kernel.__getitem__.return_value.assert_called_once_with(
                                    values,
                                    actual_values,
                                    WIDTH=width,
                                    VALUE_ROW_STRIDE=width * 4 + 7,
                                    VALUE_COLUMN_STRIDE=3,
                                    ELEMENT_BYTES=element_bytes,
                                    COPY_BLOCK=copy_tile,
                                    SPAN=1 << ((copy_tile - 1) * 3).bit_length(),
                                    multibuffer=False,
                                )
                            else:
                                actual_values = values
                                self.empty.assert_not_called()
                                self.copy_kernel.__getitem__.assert_not_called()
                            owners, index_block, copy_block = self.tiling(
                                rows,
                                width,
                                capacity,
                                element_bytes,
                                slots.element_size(),
                                actual_values.stride(1),
                                48,
                                196608,
                            )
                            self.kernel.__getitem__.assert_called_once_with((owners,))
                            self.kernel.__getitem__.return_value.assert_called_once_with(
                                cache,
                                slots,
                                actual_values,
                                N=rows,
                                WIDTH=width,
                                CACHE_ROWS=capacity,
                                CACHE_ROW_STRIDE=width * 3 + 5,
                                CACHE_COLUMN_STRIDE=2,
                                SLOT_STRIDE=3,
                                VALUE_ROW_STRIDE=actual_values.stride(0),
                                VALUE_COLUMN_STRIDE=actual_values.stride(1),
                                ELEMENT_BYTES=element_bytes,
                                OWNERS=owners,
                                INDEX_BLOCK=index_block,
                                COPY_BLOCK=copy_block,
                                multibuffer=False,
                            )
                            self.kernel.reset_mock()
                            self.copy_kernel.reset_mock()
                            self.empty.reset_mock()
        self.format_cast.assert_not_called()

    def test_empty_rows_or_width_are_noop(self):
        for function in (self.write_k, self.write_scale):
            for shape in ((0, 128), (7, 0), (0, 0)):
                cache, slots, values = Tensor((19, shape[1])), Tensor((shape[0], 1), "i32"), Tensor(shape)
                self.assertIs(function(cache, slots, values), cache)
        self.kernel.__getitem__.assert_not_called()
        self.copy_kernel.__getitem__.assert_not_called()
        self.empty.assert_not_called()
        self.format_cast.assert_not_called()
        self.resources.assert_not_called()

    def test_row_gaps_and_single_column_need_no_normalization(self):
        for width, column_stride in ((1, 0), (1, 17), (3, 1), (128, 1)):
            cache = Tensor((29, width))
            slots = Tensor((7,), "i32")
            values = Tensor((7, width), strides=(width * 3 + 5, column_stride))
            self.assertIs(self.write_k(cache, slots, values), cache)
            call = self.kernel.__getitem__.return_value.call_args
            self.assertIs(call.args[2], values)
            self.assertEqual(call.kwargs["VALUE_ROW_STRIDE"], values.stride(0))
            self.assertEqual(call.kwargs["VALUE_COLUMN_STRIDE"], column_stride)
        self.copy_kernel.__getitem__.assert_not_called()
        self.empty.assert_not_called()

    def test_transpose_dispatch_preserves_64bit_and_broadcast_span_copy(self):
        for dtype, size in (("i8", 1), ("bf16", 2), ("f32", 4), ("i64", 8)):
            for row_stride, column_stride in ((1, 131), (1, 0), (2, 131)):
                values = Tensor((128, 17), dtype, size, (row_stride, column_stride))
                cache = Tensor((1001, 17), dtype, size)
                self.write_k(cache, Tensor((128,), "i32"), values)
                if row_stride == 1 and column_stride > 1 and size <= 4:
                    first, second = self.transpose_tiling(128, 17, size, 48, 196608)
                    self.transpose_kernel.__getitem__.assert_called_once_with(
                        ((128 + first - 1) // first, (17 + second - 1) // second)
                    )
                    call = self.transpose_kernel.__getitem__.return_value.call_args
                    self.assertIs(call.args[0], values)
                    self.assertEqual(call.kwargs["VALUE_COLUMN_STRIDE"], column_stride)
                    self.assertEqual(call.kwargs["FULL"], 128 % first == 0 and 17 % second == 0)
                    self.copy_kernel.__getitem__.assert_not_called()
                else:
                    self.transpose_kernel.__getitem__.assert_not_called()
                    self.copy_kernel.__getitem__.assert_called_once()
                    call = self.copy_kernel.__getitem__.return_value.call_args
                    span = (call.kwargs["COPY_BLOCK"] - 1) * column_stride + 1
                    self.assertEqual(call.kwargs["SPAN"], 1 << (span - 1).bit_length())
                dense = self.kernel.__getitem__.return_value.call_args.args[2]
                self.assertEqual(dense._strides, (17, 1))
                self.transpose_kernel.reset_mock()
                self.copy_kernel.reset_mock()
                self.kernel.reset_mock()

    def test_transpose_tiling_tracks_padding_resources_and_parallelism(self):
        plans = []
        for rows, width, size, cores, ub in (
            (128, 17, 1, 48, 196608),
            (128, 17, 1, 8, 196608),
            (128, 17, 4, 48, 196608),
            (1025, 257, 2, 48, 196608),
            (1025, 257, 2, 48, 2048),
        ):
            first, second = self.transpose_tiling(rows, width, size, cores, ub)
            alignment = 32 // size
            aligned_first = ((first + alignment - 1) // alignment) * alignment
            aligned_second = ((second + alignment - 1) // alignment) * alignment
            working = 2 * aligned_first * aligned_second * size + (first + second) * 17
            self.assertLessEqual(working, ub)
            self.assertTrue(all(block > 0 and block & (block - 1) == 0 for block in (first, second)))
            plans.append((first, second))
        self.assertNotEqual(plans[0], plans[1])
        self.assertNotEqual(plans[-2], plans[-1])
        with self.assertRaisesRegex(RuntimeError, "Device UB"):
            self.transpose_tiling(3, 5, 1, 48, 32)

    def test_actual_transpose_copy_preserves_payload_bits_and_tail_guards(self):
        import numpy as np

        source_path = ROOT / "vllm_ascend/ops/triton/indexer_cache.py"
        kernel = next(
            node
            for node in ast.parse(source_path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_indexer_cache_transpose_kernel"
        )
        kernel.decorator_list = []

        class Index(int):
            def to(self, dtype):
                return self

        class Vector(np.ndarray):
            def to(self, dtype):
                return self.astype(dtype)

        class Pointer:
            def __init__(self, buffer, offset=0):
                self.buffer, self.offset = buffer, offset

            def __add__(self, offset):
                if isinstance(offset, np.ndarray):
                    offset = offset.astype(np.int64)
                return Pointer(self.buffer, self.offset + offset)

            def to(self, dtype):
                return Pointer(self.buffer.view(dtype), self.offset)

        def load(pointer, mask=None, other=0):
            if mask is None:
                return pointer.buffer[pointer.offset]
            result = np.full(mask.shape, other, dtype=pointer.buffer.dtype)
            result[mask] = pointer.buffer[pointer.offset[mask]]
            return result

        def store(pointer, value, mask=None):
            if mask is None:
                pointer.buffer[pointer.offset] = value
            else:
                pointer.buffer[pointer.offset[mask]] = value[mask]

        tl = SimpleNamespace(
            constexpr=None,
            int8=np.int8,
            int16=np.int16,
            int32=np.int32,
            int64=np.int64,
            pointer_type=lambda dtype: dtype,
            arange=lambda start, end: np.arange(start, end, dtype=np.int32).view(Vector),
            trans=np.transpose,
            load=load,
            store=store,
        )
        namespace = {"tl": tl}
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source_path), "exec"), namespace)
        for rows, width in ((6, 128), (128, 17), (17, 129), (33, 257)):
            for size, dtype, bits in (
                (1, np.uint8, (0, 1, 0x7F, 0x80, 0xFF)),
                (2, np.uint16, (0, 0x8000, 0x7C00, 0x7D01, 0x7FC1, 0xFFFF)),
                (4, np.uint32, (0, 0x80000000, 0x7F800000, 0x7F800001, 0x7FC00001, 0xFFFFFFFF)),
            ):
                for gap in (0, 5):
                    stride = rows + gap
                    source = np.resize(np.asarray(bits, dtype=dtype), width * stride + 7)
                    before = source.copy()
                    target = np.full(rows * width + 11, 0x5A, dtype=dtype)
                    expected = np.asarray(
                        [[source[3 + column * stride + row] for column in range(width)] for row in range(rows)],
                        dtype=dtype,
                    ).reshape(-1)
                    first, second = self.transpose_tiling(rows, width, size, 48, 196608)
                    for row_tile in range((rows + first - 1) // first):
                        for column_tile in range((width + second - 1) // second):
                            tl.program_id = lambda axis, row_tile=row_tile, column_tile=column_tile: Index(
                                row_tile if axis == 0 else column_tile
                            )
                            namespace[kernel.name](
                                Pointer(source, 3),
                                Pointer(target),
                                rows,
                                width,
                                stride,
                                size,
                                first,
                                second,
                                rows % first == 0 and width % second == 0,
                            )
                    np.testing.assert_array_equal(target[: rows * width], expected)
                    np.testing.assert_array_equal(target[rows * width :], 0x5A)
                    np.testing.assert_array_equal(source, before)

        # Probe address arithmetic without allocating enormous tensors. The
        # source/target local axes must widen independently of scalar origins.
        for width, stride, row_tile, column_tile in (
            (128, 64, 1 << 28, 0),
            (1 << 32, 1 << 36, 3, 2),
            (128, 1 << 36, 3, 2),
            (1 << 32, 64, 3, 2),
        ):
            addresses = {}

            def record_load(pointer, addresses=addresses):
                addresses["source"] = pointer.offset
                return np.zeros((8, 8), dtype=np.int8)

            def record_store(pointer, value, addresses=addresses):
                addresses["target"] = pointer.offset

            tl.load, tl.store = record_load, record_store
            tl.program_id = lambda axis, row_tile=row_tile, column_tile=column_tile: Index(
                row_tile if axis == 0 else column_tile
            )
            dummy = np.zeros(1, dtype=np.uint8)
            namespace[kernel.name](Pointer(dummy), Pointer(dummy), 1 << 34, width, stride, 1, 8, 8, True)
            expected_source = np.asarray(
                [[(column_tile * 8 + column) * stride + row_tile * 8 + row for row in range(8)] for column in range(8)],
                dtype=np.int64,
            )
            expected_target = np.asarray(
                [[(row_tile * 8 + row) * width + column_tile * 8 + column for column in range(8)] for row in range(8)],
                dtype=np.int64,
            )
            np.testing.assert_array_equal(addresses["source"], expected_source)
            np.testing.assert_array_equal(addresses["target"], expected_target)

    def test_tiling_tracks_work_width_dtype_stride_and_device_budget(self):
        self.assertEqual(self.tiling(6, 128, 1000000, 1, 4, 1, 48, 196608), (6, 8, 128))
        self.assertEqual(self.tiling(6, 1, 1000000, 2, 8, 1, 48, 196608), (6, 8, 1))
        self.assertEqual(self.tiling(100, 3, 7, 1, 4, 1, 48, 196608)[0], 7)
        self.assertEqual(self.tiling(100, 3, 1000, 1, 4, 1, 24, 196608)[0], 24)
        self.assertGreater(self.tiling(6, 100000, 1000, 8, 8, 2, 48, 196608)[2], 1)
        self.assertLess(
            self.tiling(6, 100000, 1000, 8, 8, 17, 48, 196608)[2], self.tiling(6, 100000, 1000, 8, 8, 2, 48, 196608)[2]
        )
        plans = [
            self.tiling(65537, 65537, 1000000, size, index_size, 1, cores, ub)
            for size, index_size, cores, ub in ((1, 4, 48, 196608), (8, 8, 48, 196608), (8, 8, 24, 32768))
        ]
        self.assertGreater(plans[0][1] * plans[0][2], plans[1][1] * plans[1][2])
        self.assertGreater(plans[1][1] * plans[1][2], plans[2][1] * plans[2][2])
        for plan, sizes, budget in zip(plans, ((1, 4), (8, 8), (8, 8)), (196608, 196608, 32768)):
            _, index_block, copy_block = plan
            buffers = [(index_block, size) for size in (sizes[1], 8, 8, 1)]
            buffers += [(copy_block, size) for size in (sizes[0], 8, 8, 1)]
            self.assertLessEqual(sum(((lanes * size + 31) // 32) * 32 for lanes, size in buffers), budget)
            self.assertTrue(all(block > 0 and block & (block - 1) == 0 for block in (index_block, copy_block)))
        with self.assertRaisesRegex(RuntimeError, "Device UB"):
            self.tiling(1, 1, 1, 8, 8, 1, 1, 32)

    def test_two_writes_remain_two_independent_launches(self):
        slots = Tensor((17,), "i64", strides=(2,))
        k_cache, scale_cache = Tensor((101, 129)), Tensor((101, 3), "f32", 4)
        k, scale = Tensor((17, 129)), Tensor((17, 3), "f32", 4)
        self.assertIs(self.write_k(k_cache, slots, k), k_cache)
        self.assertIs(self.write_scale(scale_cache, slots, scale), scale_cache)
        calls = self.kernel.__getitem__.return_value.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.args for call in calls], [(k_cache, slots, k), (scale_cache, slots, scale)])

    def test_actual_owner_loop_covers_slots_once_in_input_order_across_tiles(self):
        """Execute the real ownership/index loop; device store ordering needs the NPU gate."""
        source = ROOT / "vllm_ascend/ops/triton/indexer_cache.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_indexer_cache_scatter_kernel"
        )

        class RecordCopy(ast.NodeTransformer):
            def visit_For(self, node):
                if isinstance(node.target, ast.Name) and node.target.id == "begin":
                    return ast.parse("record(row, slot)").body[0]
                return self.generic_visit(node)

        kernel = RecordCopy().visit(kernel)
        kernel.decorator_list = []
        for argument in kernel.args.args:
            argument.annotation = None

        class Index(int):
            def to(self, dtype):
                return self

        class Vector(list):
            def to(self, dtype):
                return self

            def __add__(self, other):
                return Vector(value + other for value in self)

            __radd__ = __add__

            def __mul__(self, other):
                return Vector(value * other for value in self)

            def __lt__(self, other):
                return Vector(value < other for value in self)

        class Pointer:
            def __init__(self, values, offsets=None):
                self.values, self.offsets = values, offsets

            def __add__(self, offsets):
                return Pointer(self.values, offsets)

            def to(self, dtype):
                return self

        for rows, capacity, cores, ub in (
            (1, 1, 48, 196608),
            (17, 7, 48, 512),
            (129, 31, 24, 1024),
            (257, 1000000, 48, 1024),
        ):
            for slot_stride in (0, 1, 3):
                owners, index_block, copy_block = self.tiling(rows, 3, capacity, 1, 8, 1, cores, ub)
                slots = [((row * 7) % capacity) for row in range(rows)]
                slots[0] = slots[-1] = capacity - 1
                if rows > 3:
                    slots[1:4] = [-1, -(1 << 40), capacity + (1 << 40)]
                backing = [-1] * (rows * max(slot_stride, 1))
                for row, slot in enumerate(slots):
                    backing[row * max(slot_stride, 1)] = slot
                logical = [backing[row * slot_stride] for row in range(rows)]
                records = []

                def load(pointer, mask, other):
                    return Vector(
                        Index(pointer.values[offset] if valid else other)
                        for offset, valid in zip(pointer.offsets, mask)
                    )

                tl = SimpleNamespace(
                    constexpr=None,
                    int64=None,
                    uint8=None,
                    uint16=None,
                    uint32=None,
                    uint64=None,
                    minimum=min,
                    static_assert=lambda condition: self.assertTrue(condition),
                    pointer_type=lambda raw_type: raw_type,
                    range=lambda *args: map(Index, range(*args)),
                    arange=lambda first, end: Vector(range(first, end)),
                    load=MagicMock(side_effect=load),
                )
                namespace = dict(
                    tl=tl,
                    get_element=lambda vector, indices: vector[indices[0]],
                    record=lambda row, slot, records=records: records.append((int(row), int(slot))),
                )
                module = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
                exec(compile(module, str(source), "exec"), namespace)
                for owner in reversed(range(owners)):
                    tl.program_id = lambda axis, owner=owner: Index(owner)
                    namespace[kernel.name](
                        Pointer([]),
                        Pointer(backing),
                        Pointer([]),
                        rows,
                        3,
                        capacity,
                        3,
                        1,
                        slot_stride,
                        3,
                        1,
                        1,
                        owners,
                        index_block,
                        copy_block,
                    )
                expected_rows = [row for row, slot in enumerate(logical) if 0 <= slot < capacity]
                self.assertEqual(sorted(row for row, slot in records), expected_rows)
                for slot in set(logical):
                    self.assertEqual(
                        [row for row, written in records if written == slot],
                        [row for row in expected_rows if logical[row] == slot],
                    )
                self.assertEqual(tl.load.call_count, owners * ((rows + index_block - 1) // index_block))

    def test_actual_normalization_preserves_bits_and_source_holes(self):
        source = ROOT / "vllm_ascend/ops/triton/indexer_cache.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_indexer_cache_copy_kernel"
        )
        code = compile(ast.Module(body=kernel.body, type_ignores=[]), str(source), "exec")

        class Index(int):
            def to(self, dtype):
                return self

        class Vector(list):
            def to(self, dtype, bitcast=False):
                if dtype != "int64":
                    self_test.assertTrue(bitcast, "Payload conversion must preserve raw bits")
                return self

            def __add__(self, other):
                if isinstance(other, list):
                    return Vector(value + addition for value, addition in zip(self, other))
                return Vector(value + other for value in self)

            __radd__ = __add__

            def __mul__(self, other):
                return Vector(value * other for value in self)

            def __floordiv__(self, other):
                return Vector(value // other for value in self)

            def __mod__(self, other):
                return Vector(value % other for value in self)

            def __lt__(self, other):
                return Vector(value < other for value in self)

        class Pointer:
            def __init__(self, buffer, offset=0, word_bytes=None):
                self.buffer, self.offset = buffer, offset
                self.word_bytes = element_bytes if word_bytes is None else word_bytes

            def __add__(self, offset):
                return Pointer(self.buffer, self.offset + offset, self.word_bytes)

            def to(self, dtype):
                sizes = {"uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8}
                size = sizes[dtype]
                return Pointer(self.buffer, self.offset * self.word_bytes // size, size)

        self_test = self

        for width in (1, 31, 128, 129, 257):
            for source_stride in (0, 2, 3, 17, 129):
                for element_bytes in (1, 2, 4, 8):
                    with self.subTest(width=width, source_stride=source_stride, element_bytes=element_bytes):
                        mask = (1 << (element_bytes * 8)) - 1
                        source_row_stride = width * max(source_stride, 1) + 5
                        values = [
                            (index * 0x9E3779B97F4A7C15 + (index >> 8)) & mask for index in range(3 * source_row_stride)
                        ]
                        before_values = values[:]
                        output = [0x5A & mask] * (3 * width)
                        expected = output[:]
                        for column in range(width):
                            expected[2 * width + column] = values[2 * source_row_stride + column * source_stride]
                        _, _, copy_block = self.tiling(3, width, 7, element_bytes, 8, source_stride, 7, 1024)

                        def load(
                            pointer,
                            mask,
                            other,
                            source_row_stride=source_row_stride,
                            element_bytes=element_bytes,
                            width=width,
                            source_stride=source_stride,
                        ):
                            valid_offsets = [offset for offset, valid in zip(pointer.offset, mask) if valid]
                            # Physical-span reads must remain within this source row's actual last element.
                            self.assertTrue(
                                all(
                                    2 * source_row_stride * element_bytes
                                    <= offset * pointer.word_bytes
                                    <= (2 * source_row_stride + (width - 1) * source_stride + 1) * element_bytes
                                    - pointer.word_bytes
                                    for offset in valid_offsets
                                )
                            )
                            return Vector(
                                (
                                    (
                                        pointer.buffer[offset * pointer.word_bytes // element_bytes]
                                        >> (8 * (offset * pointer.word_bytes % element_bytes))
                                    )
                                    & ((1 << (8 * pointer.word_bytes)) - 1)
                                )
                                if valid
                                else other
                                for offset, valid in zip(pointer.offset, mask)
                            )

                        def store(pointer, value, mask, element_bytes=element_bytes):
                            for offset, item, valid in zip(pointer.offset, value, mask):
                                if valid:
                                    position = offset * pointer.word_bytes // element_bytes
                                    shift = 8 * (offset * pointer.word_bytes % element_bytes)
                                    word_mask = ((1 << (8 * pointer.word_bytes)) - 1) << shift
                                    pointer.buffer[position] = (pointer.buffer[position] & ~word_mask) | (item << shift)

                        tl = SimpleNamespace(
                            range=MagicMock(side_effect=lambda *args: map(Index, range(*args))),
                            arange=lambda first, last: Vector(range(first, last)),
                            minimum=min,
                            gather=lambda value, indices, axis: Vector(value[index] for index in indices),
                            load=MagicMock(side_effect=load),
                            store=MagicMock(side_effect=store),
                            static_range=range,
                            pointer_type=lambda dtype: dtype,
                            int8="int8",
                            int64="int64",
                            uint8="uint8",
                            uint16="uint16",
                            uint32="uint32",
                            uint64="uint64",
                            float16="float16",
                            float32="float32",
                            constexpr=None,
                        )
                        namespace = dict(
                            tl=tl,
                            WIDTH=width,
                            ELEMENT_BYTES=element_bytes,
                            COPY_BLOCK=copy_block,
                            SPAN=1 << ((copy_block - 1) * source_stride).bit_length(),
                            VALUE_COLUMN_STRIDE=source_stride,
                            VALUE_ROW_STRIDE=source_row_stride,
                            values=Pointer(values),
                            output=Pointer(output),
                        )
                        for tile in range((width + copy_block - 1) // copy_block):
                            tl.program_id = lambda axis, tile=tile: Index(2 if axis == 0 else tile)
                            exec(code, namespace)
                        self.assertEqual(output, expected)
                        self.assertEqual(values, before_values)
                        expected_copies = (width + copy_block - 1) // copy_block
                        self.assertEqual(tl.load.call_count, expected_copies)
                        self.assertEqual(tl.store.call_count, expected_copies)

    def test_basic_scatter_contract_rejects_mismatch_before_device_work(self):
        invalid = [
            (0, Tensor((101,))),
            (2, Tensor((17,))),
            (0, Tensor((101, 3))),
            (2, Tensor((17, 129), "f16", 2)),
            (1, Tensor((16,), "i32")),
            (1, Tensor((17, 2), "i32")),
            (1, Tensor((17,), "i16")),
        ]
        for position, shape, dtype in ((0, (101, 129), "i8"), (1, (17,), "i32"), (2, (17, 129), "i8")):
            invalid.extend(
                (position, Tensor(shape, dtype, **change)) for change in ({"device": "cpu"}, {"device_index": 1})
            )
        for function in (self.write_k, self.write_scale):
            for position, value in invalid:
                with self.subTest(function=function.__name__, position=position, shape=value.shape, dtype=value.dtype):
                    args = [Tensor((101, 129)), Tensor((17,), "i32"), Tensor((17, 129))]
                    args[position] = value
                    with self.assertRaises(ValueError):
                        function(*args)
        self.kernel.__getitem__.assert_not_called()
        self.get_format.assert_not_called()
        self.format_cast.assert_not_called()

    def test_each_non_nd_operand_is_rejected_before_conversion_launch_or_copyback(self):
        for function in (self.write_k, self.write_scale):
            for rows, width in ((5, 7), (0, 7), (5, 0)):
                for position in range(3):
                    for fmt in (29, 0):
                        with self.subTest(
                            function=function.__name__, shape=(rows, width), position=position, format=fmt
                        ):
                            cache = Tensor((19, width), "bf16", 2, (width * 3 + 5, 2))
                            slots = Tensor((rows, 1), "i32", strides=(3, 1))
                            values = Tensor((rows, width), "bf16", 2, (width * 4 + 7, 3))
                            operands = (cache, slots, values)
                            operands[position].format = fmt
                            for tensor in operands:
                                tensor.copy_ = MagicMock()
                            with self.assertRaisesRegex(RuntimeError, "requires ND tensors"):
                                function(*operands)
                            self.format_cast.assert_not_called()
                            self.kernel.__getitem__.assert_not_called()
                            for tensor in operands:
                                tensor.copy_.assert_not_called()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path)
    args, remaining = parser.parse_known_args()
    if args.source_root is not None:
        ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
