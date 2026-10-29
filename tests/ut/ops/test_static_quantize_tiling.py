# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of CANN core rectangles, independent UB tiles and ND addressing."""

import argparse
import ast
import itertools
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/static_quantize.py"


def load_host_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_cann_ub_bytes",
            "_cann_ub_factors",
            "_quantize_ub_bytes",
            "_quantize_tiling",
            "_quantize_contiguous_copy",
            "quantize",
        )
    ]
    namespace = {
        "DMA_ALIGNMENT_BYTES": 32,
        "torch": SimpleNamespace(Tensor=object),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def covered_indices(plan, rows, width):
    for core in range(plan["NUM_CORES"]):
        row_start = core // plan["COLUMN_CORES"] * plan["ROW_FACTOR"]
        column_start = core % plan["COLUMN_CORES"] * plan["COLUMN_FACTOR"]
        row_end = min(row_start + plan["ROW_FACTOR"], rows)
        column_end = min(column_start + plan["COLUMN_FACTOR"], width)
        for column_tile in range(0, plan["COLUMN_FACTOR"], plan["BLOCK_COLS"]):
            for row_batch in range(row_start, row_end, plan["BASE_N"]):
                for row in range(row_batch, min(row_batch + plan["BASE_N"], row_end)):
                    for column in range(
                        column_start + column_tile, min(column_start + column_tile + plan["BLOCK_COLS"], column_end)
                    ):
                        yield row * width + column


class QuantizeTilingTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_host_functions()
        self.plan = self.namespace["_quantize_tiling"]
        self.budget = self.namespace["_cann_ub_bytes"]
        self.working_set = self.namespace["_quantize_ub_bytes"]

    def make_plan(
        self,
        rows,
        width,
        element_bytes=2,
        parameter_bytes=2,
        contiguous=True,
        parameter_contiguous=True,
        per_tensor=False,
        cores=48,
        ub=196608,
        line=512,
    ):
        return self.plan(
            rows, width, element_bytes, parameter_bytes, contiguous, parameter_contiguous, per_tensor, cores, ub, line
        )

    def test_strided_layouts_keep_original_cann_core_partition(self):
        # CANN v9.1 source / trace: hardware pool 48 gives 6 rows x 8
        # column cores for these shapes. This is an original-source fixture,
        # not a model-specific policy in the production planner.
        for width in (8192, 2048):
            for contiguous, parameter_contiguous in itertools.product((False, True), repeat=2):
                if contiguous and parameter_contiguous:
                    continue
                original = self.make_plan(6, width, contiguous=contiguous, parameter_contiguous=parameter_contiguous)
                self.assertEqual(
                    (
                        original["NUM_CORES"],
                        original["ROW_FACTOR"],
                        original["COLUMN_CORES"],
                        original["COLUMN_FACTOR"],
                    ),
                    (48, 1, 8, width // 8),
                )
                self.assertEqual(
                    (original["BASE_N"], original["BLOCK_ROWS"], original["BLOCK_COLS"]), (1, 1, width // 8)
                )
        row = self.make_plan(67, 17)
        self.assertEqual((row["NUM_CORES"], row["ROW_FACTOR"], row["COLUMN_CORES"]), (34, 2, 1))
        column = self.make_plan(1, 4097, contiguous=False)
        self.assertEqual((column["NUM_CORES"], column["ROW_FACTOR"], column["COLUMN_CORES"]), (17, 1, 17))
        legacy = self.make_plan(10, 17, line=256)
        self.assertEqual((legacy["NUM_CORES"], legacy["ROW_FACTOR"]), (5, 2))
        self.assertEqual(set(covered_indices(legacy, 10, 17)), set(range(170)))

    def test_small_contiguous_rows_use_resource_sized_tasks(self):
        # Expected consequences of the measured 196608B UB / 48-core resource
        # plan, not shape guards. A wider row requires more UB-sized pieces.
        for rows, width, cores, column_factor in (
            (6, 8192, 6, 8192),
            (6, 2048, 6, 2048),
            (7, 16384, 14, 8192),
            (1, 65536, 8, 8192),
        ):
            plan = self.make_plan(rows, width)
            self.assertEqual((plan["NUM_CORES"], plan["COLUMN_FACTOR"]), (cores, column_factor))
            self.assertEqual(sorted(covered_indices(plan, rows, width)), list(range(rows * width)))
        # Capacity, rather than an element-count constant, changes the pool.
        normal = self.make_plan(3, 16384)
        larger_ub = self.make_plan(3, 16384, ub=393216)
        self.assertEqual((normal["NUM_CORES"], larger_ub["NUM_CORES"]), (6, 3))
        self.assertGreater(larger_ub["BLOCK_COLS"], normal["BLOCK_COLS"])
        for cores in (7, 13, 48):
            plan = self.make_plan(5, 8193, cores=cores)
            self.assertLessEqual(plan["NUM_CORES"], cores)
            self.assertEqual(sorted(covered_indices(plan, 5, 8193)), list(range(5 * 8193)))

    def test_large_contiguous_plans_remain_equal_to_test064_baseline(self):
        # Full plan fixtures from source 029a3dc / TEST-064. Protect the
        # already-measured wide-row and multirow paths, including FP32.
        for dtype_bytes in (2, 4):
            for rows, cores, row_factor in ((48, 48, 1), (128, 43, 3), (512, 47, 11)):
                self.assertEqual(
                    self.make_plan(rows, 8192, element_bytes=dtype_bytes, parameter_bytes=dtype_bytes),
                    {
                        "NUM_CORES": cores,
                        "ROW_FACTOR": row_factor,
                        "COLUMN_FACTOR": 8192,
                        "COLUMN_CORES": 1,
                        "BASE_N": 1,
                        "BLOCK_ROWS": 1,
                        "BLOCK_COLS": 8192,
                    },
                )
            self.assertEqual(
                self.make_plan(513, 17, element_bytes=dtype_bytes, parameter_bytes=dtype_bytes),
                {
                    "NUM_CORES": 47,
                    "ROW_FACTOR": 11,
                    "COLUMN_FACTOR": 17,
                    "COLUMN_CORES": 1,
                    "BASE_N": 11,
                    "BLOCK_ROWS": 16,
                    "BLOCK_COLS": 512 // dtype_bytes,
                },
            )

    def test_scalar_partition_uses_flat_input_and_hardware_pool(self):
        scalar = self.make_plan(1, 49152, parameter_bytes=4, parameter_contiguous=False, per_tensor=True)
        self.assertEqual((scalar["NUM_CORES"], scalar["ROW_FACTOR"], scalar["COLUMN_FACTOR"]), (48, 1, 1024))
        self.assertEqual(sorted(covered_indices(scalar, 1, 49152)), list(range(49152)))

    def test_cann_dtype_specific_buffers_not_one_bytes_per_element_constant(self):
        # Independent InitBuffer totals from the v9.1 FP16/BF16/FP32 headers.
        # BF16/FP16: batch x/y + one FP32 work row + raw/FP32 parameters.
        self.assertEqual(self.budget(3, 256, 2, 2, False), (3 * 3 + 4 + 12) * 256)
        # FP32: batch input is reused as its work buffer, parameters need no cast.
        self.assertEqual(self.budget(3, 256, 4, 4, False), (3 * 5 + 8) * 256)
        # Scalar BF16 parameters occupy two 32B buffers; FP32 reads GM scalars.
        self.assertEqual(self.budget(1, 256, 2, 2, True), 7 * 256 + 64)
        self.assertEqual(self.budget(1, 256, 4, 4, True), 5 * 256)

    def test_original_calc_ub_factor_reference(self):
        factors = self.namespace["_cann_ub_factors"]
        # These are evaluated from the original CANN formulas with its actual
        # 196608-byte UB and 2048-byte reserve, before Triton carrier rounding.
        for dtype_bytes, per_tensor, expected in (
            (2, False, 10240),
            (4, False, 14848),
            (2, True, 27648),
            (4, True, 38912),
        ):
            self.assertEqual(factors(1, 1 << 20, dtype_bytes, per_tensor, False, 196608, 512), (1, expected))
        self.assertEqual(factors(11, 17, 2, False, False, 196608, 512), (11, 256))
        self.assertEqual(factors(3, 8192, 2, False, False, 196608, 512), (1, 8192))
        self.assertEqual(factors(1, 1024, 2, False, True, 196608, 512), (1, 1024))

    def test_smaller_ub_adds_tiles_without_changing_core_ownership(self):
        large = self.make_plan(257, 16387, cores=7)
        small = self.make_plan(257, 16387, cores=7, ub=8192)
        for field in ("NUM_CORES", "ROW_FACTOR", "COLUMN_FACTOR", "COLUMN_CORES"):
            self.assertEqual(large[field], small[field])
        self.assertLess(small["BLOCK_COLS"], large["BLOCK_COLS"])
        self.assertGreater(small["COLUMN_FACTOR"], small["BLOCK_COLS"])
        many_rows = self.make_plan(513, 17, cores=1)
        self.assertEqual(many_rows["ROW_FACTOR"], 513)
        self.assertEqual(set(covered_indices(many_rows, 513, 17)), set(range(513 * 17)))
        self.assertGreater(many_rows["BASE_N"], 1)
        self.assertGreaterEqual(many_rows["BLOCK_ROWS"], many_rows["BASE_N"])
        with self.assertRaisesRegex(RuntimeError, "UB cannot hold"):
            self.make_plan(1, 17, ub=1)

    def test_batch_carrier_padding_does_not_exceed_cann_buffer_plan(self):
        plan = self.make_plan(513, 17)
        self.assertEqual((plan["NUM_CORES"], plan["ROW_FACTOR"], plan["BASE_N"]), (47, 11, 11))
        self.assertEqual((plan["BLOCK_ROWS"], plan["BLOCK_COLS"]), (16, 256))
        for dtype_bytes, width in itertools.product((2, 4), (17, 257, 8193, 1 << 20)):
            plan = self.make_plan(513, width, element_bytes=dtype_bytes, parameter_bytes=dtype_bytes)
            self.assertLessEqual(
                self.budget(plan["BLOCK_ROWS"], plan["BLOCK_COLS"], dtype_bytes, dtype_bytes, False), 196608 - 2048
            )

    def test_core_and_ub_tails_cover_every_output_once(self):
        cases = ((1, 31), (2, 33), (3, 257), (7, 1001), (67, 17), (5, 8193), (1, 65539))
        for rows, width in cases:
            for cores, element_bytes, contiguous, parameter_contiguous, ub in itertools.product(
                (1, 7, 48), (2, 4), (True, False), (True, False), (8192, 196608)
            ):
                with self.subTest(
                    rows=rows,
                    width=width,
                    cores=cores,
                    dtype_bytes=element_bytes,
                    contiguous=contiguous,
                    parameter_contiguous=parameter_contiguous,
                    ub=ub,
                ):
                    scalar = rows == 1
                    plan = self.make_plan(
                        rows, width, element_bytes, element_bytes, contiguous, parameter_contiguous, scalar, cores, ub
                    )
                    covered = list(covered_indices(plan, rows, width))
                    self.assertEqual(len(covered), rows * width)
                    self.assertEqual(set(covered), set(range(rows * width)))
                    self.assertLessEqual(plan["NUM_CORES"], cores)
                    self.assertGreater(plan["NUM_CORES"], 0)
                    self.assertEqual(plan["BLOCK_COLS"] & (plan["BLOCK_COLS"] - 1), 0)
                    self.assertEqual(plan["BLOCK_COLS"] % 32, 0)
                    self.assertLessEqual(
                        self.budget(plan["BLOCK_ROWS"], plan["BLOCK_COLS"], element_bytes, element_bytes, scalar),
                        ub - 2048,
                    )
                    self.assertLessEqual(
                        self.working_set(
                            plan["BLOCK_ROWS"],
                            plan["BLOCK_COLS"],
                            element_bytes,
                            element_bytes,
                            contiguous,
                            parameter_contiguous,
                            scalar,
                        ),
                        ub - 2048,
                    )

    def test_strided_nd_addresses_match_coordinate_definition(self):
        for shape, strides in (((2, 3, 17), (211, 37, 2)), ((3, 5), (0, 2)), ((2, 3, 4), (1, 8, 2))):
            expected = {
                sum(position * stride for position, stride in zip(coordinates, strides))
                for coordinates in itertools.product(*(range(dimension) for dimension in shape))
            }
            for scalar in (False, True):
                width = math.prod(shape) if scalar else shape[-1]
                rows = 1 if scalar else math.prod(shape[:-1])
                plan = self.make_plan(
                    rows, width, contiguous=False, parameter_contiguous=False, per_tensor=scalar, ub=8192
                )
                actual = set()
                for index in covered_indices(plan, rows, width):
                    remaining, physical = index, 0
                    for dimension, stride in zip(reversed(shape), reversed(strides)):
                        physical += remaining % dimension * stride
                        remaining //= dimension
                    actual.add(physical)
                self.assertEqual(actual, expected)

    def test_parameter_reads_are_outside_the_row_reuse_loop(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_static_quantize_kernel"
        )
        columns = next(node for node in kernel.body if isinstance(node, ast.For) and node.target.id == "column_tile")
        row_loop = next(node for node in columns.body if isinstance(node, ast.For) and node.target.id == "row_batch")
        names = {node.id for node in ast.walk(row_loop) if isinstance(node, ast.Name)}
        self.assertTrue({"reciprocal", "offset"} <= names)
        self.assertFalse({"reciprocal_ptr", "offset_ptr"} & names)
        parameter_branch = next(node for node in columns.body if isinstance(node, ast.If))
        self.assertFalse(any(isinstance(node, ast.Mod) for node in ast.walk(parameter_branch)))

    def test_batch_uses_runtime_rows_half_carrier_and_one_final_cast(self):
        # The A3 probe established dynamic 2D slices, 1D row computation and
        # FP16 insertion. INT8 insertion converted the entire batch per row;
        # static row expansion allocated separate workspaces for each row.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_static_quantize_kernel"
        )
        row_loop = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "row"
        )
        self.assertEqual(ast.unparse(row_loop.iter), "range(batch_end - row_batch)")
        self.assertNotIn("tl.int8", ast.unparse(row_loop))
        self.assertNotIn("tl.store", ast.unparse(row_loop))
        input_branch = next(node for node in row_loop.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(input_branch.test), "CONTIGUOUS")
        self.assertNotIn("tl.load", ast.unparse(ast.Module(body=input_branch.body, type_ignores=[])))
        self.assertIn("_load_quantize_columns", ast.unparse(ast.Module(body=input_branch.orelse, type_ignores=[])))
        self.assertIn("al.extract_slice", ast.unparse(row_loop))
        self.assertIn("al.insert_slice", ast.unparse(row_loop))
        carrier = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "batch_y"
        )
        self.assertEqual(ast.unparse(carrier.value.args[-1]), "tl.float16")
        compute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_quantize_row")
        self.assertIn("rounded = bounded + RNE_MAGIC - RNE_MAGIC", ast.unparse(compute))
        # The row result must retain the exact FP16 integer carrier; an
        # unnecessary intermediate INT32 conversion is not part of the contract.
        self.assertEqual(ast.unparse(compute.body[-1].value.args[0]), "tl.float16")

    def test_strided_row_load_addresses_match_the_original_view(self):
        # Check the address actually passed to the load helper in both the
        # single-row and batched paths, not an unused intermediate expression.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_static_quantize_kernel"
        )
        single = next(
            node for node in ast.walk(kernel) if isinstance(node, ast.If) and ast.unparse(node.test) == "BASE_N == 1"
        )
        layout = next(node for node in single.body if isinstance(node, ast.If))
        load = next(
            node
            for node in single.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "not CONTIGUOUS and (not PER_TENSOR)"
        )
        row_loop = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "row"
        )
        branch = next(node for node in row_loop.body if isinstance(node, ast.If))
        paths = {
            "single": layout.orelse[0].body + load.body,
            "batch": branch.orelse,
        }

        class Shape(tuple):
            @property
            def value(self):
                return tuple(self)

        for shape, strides in (
            ((67, 17), (37, 2)),
            ((2, 3, 5), (1, 14, 2)),
            ((3, 4, 7), (0, 19, 2)),
            ((4, 3, 2), (11, 1, 0)),
        ):
            for name, body in paths.items():
                address_code = compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec")
                for coordinates in itertools.product(*(range(dimension) for dimension in shape)):
                    logical_row = 0
                    for coordinate, dimension in zip(coordinates[:-1], shape[:-1]):
                        logical_row = logical_row * dimension + coordinate
                    loaded = MagicMock(side_effect=lambda ptr, start, end, stride, *rest: ptr + start * stride)
                    environment = {
                        "tl": SimpleNamespace(full=lambda *args: 0, int64=object(), static_range=range),
                        "SHAPE": Shape(shape),
                        "STRIDES": SimpleNamespace(value=strides),
                        "row_batch": logical_row if name == "single" else logical_row // 2 * 2,
                        "row": logical_row % 2,
                        "x_ptr": 0,
                        "column_start": 0,
                        "column_tile": coordinates[-1],
                        "column_end": shape[-1],
                        "BLOCK_COLS": 1,
                        "DMA_BYTES": 32,
                        "INPUT_BYTES": 2,
                        "INPUT_SPAN": 1,
                        "FULL_COLUMNS": True,
                        "_load_quantize_columns": loaded,
                    }
                    exec(address_code, environment)
                    loaded.assert_called_once()
                    self.assertEqual(
                        environment["x"],
                        sum(coordinate * stride for coordinate, stride in zip(coordinates, strides)),
                        (name, shape, strides, coordinates),
                    )
                    self.assertEqual(loaded.call_args.args[2], shape[-1])

    def test_working_set_accounts_for_batch_carrier_and_strided_dma(self):
        columns = 256
        for rows in (1, 4, 16):
            contiguous = self.working_set(rows, columns, 2, 2, True, True, False)
            strided_parameters = self.working_set(rows, columns, 2, 2, True, False, False)
            strided_input = self.working_set(rows, columns, 2, 2, False, True, False)
            # Strided parameter lowering stages one 32B block plus an address
            # per lane; input staging scales with the whole input carrier.
            self.assertGreaterEqual(strided_parameters - contiguous, 2 * columns * (8 + 32))
            self.assertGreaterEqual(strided_input - contiguous, 3 * columns * 8 + rows * columns * 32)
            self.assertGreater(contiguous, self.budget(rows, columns, 2, 2, False))

    def test_affine_row_budget_tracks_reused_physical_buffers(self):
        # Frozen A3 physical IR at 4096 BF16 lanes: last FP32 work row ends
        # at 90624 bytes; vsel scratch is 64 bytes. The source bound includes
        # an additional output row and the separate CANN reserve.
        actual_extent = 90624 + 64
        self.assertGreaterEqual(self.working_set(1, 4096, 2, 2, True, True, False), actual_extent)
        for element_bytes in (2, 4):
            small = self.make_plan(128, 8192, element_bytes=element_bytes, parameter_bytes=element_bytes, ub=32768)
            large = self.make_plan(128, 8192, element_bytes=element_bytes, parameter_bytes=element_bytes)
            self.assertLess(small["BLOCK_COLS"], large["BLOCK_COLS"])
            self.assertEqual(small["NUM_CORES"], large["NUM_CORES"])
            for plan, capacity in ((small, 32768), (large, 196608)):
                self.assertLessEqual(
                    self.working_set(
                        plan["BLOCK_ROWS"], plan["BLOCK_COLS"], element_bytes, element_bytes, True, True, False
                    ),
                    capacity - 2048,
                )

    def test_launcher_preserves_parameter_strides_scalar_and_empty_inputs(self):
        kernel = MagicMock()
        resources = MagicMock(return_value=SimpleNamespace(vector_cores=7, ub_bytes=8192, cache_line_bytes=512))
        output, device = object(), SimpleNamespace(type="npu", index=3)

        def tensor(shape, dtype="bf16", strides=None, contiguous=False, npu_format=2):
            strides = strides if strides is not None else ((1,) if len(shape) else ())
            return SimpleNamespace(
                dtype=dtype,
                layout="strided",
                ndim=len(shape),
                device=device,
                shape=shape,
                numel=lambda: math.prod(shape),
                element_size=lambda: 4 if dtype == "f32" else 2,
                is_contiguous=lambda: contiguous,
                stride=lambda axis=None: strides if axis is None else strides[axis],
                npu_format=npu_format,
            )

        self.namespace.update(
            torch=SimpleNamespace(
                Tensor=object,
                bfloat16="bf16",
                float16="f16",
                float32="f32",
                int8="i8",
                strided="strided",
                empty=MagicMock(return_value=output),
            ),
            torch_npu=SimpleNamespace(get_npu_format=lambda value: value.npu_format),
            get_device_resources=resources,
            _static_quantize_kernel=kernel,
        )
        quantize = self.namespace["quantize"]
        x = tensor((2, 3, 17), strides=(211, 37, 2))
        reciprocal, offset = tensor((17,), strides=(2,)), tensor((17,), strides=(0,))
        self.assertIs(quantize(x, reciprocal, offset), output)
        arguments = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual((arguments["ROW_COUNT"], arguments["COLUMN_COUNT"], arguments["PER_TENSOR"]), (6, 17, False))
        self.assertEqual((arguments["RECIPROCAL_STRIDE"], arguments["OFFSET_STRIDE"]), (2, 0))
        self.assertEqual(arguments["STRIDES"], (211, 37, 2))
        self.assertFalse(arguments["enable_fp_fusion"])
        self.assertFalse(arguments["multibuffer"])
        self.assertTrue(arguments["enable_ubuf_saving"])
        self.assertEqual(arguments["PARAMETER_BLOCK"], 16)
        for name, stride in (("INPUT_SPAN", 2), ("RECIPROCAL_SPAN", 2), ("OFFSET_SPAN", 0)):
            span = arguments[name]
            extent = (arguments["BLOCK_COLS"] - 1) * stride + 1
            self.assertIs(type(span), int)
            self.assertEqual(span & (span - 1), 0)
            self.assertGreaterEqual(span, extent)
            self.assertLess(span, 2 * extent)
        normalized = tensor((17, 129), strides=(129, 1), contiguous=True)
        normalize = MagicMock(return_value=normalized)
        self.namespace["_quantize_contiguous_copy"] = normalize
        transposed = tensor((17, 129), strides=(1, 17))
        channel = tensor((129,))
        self.assertIs(quantize(transposed, channel, channel), output)
        normalize.assert_called_once()
        self.assertEqual(normalize.call_args.args[1], 0)
        arguments = kernel.__getitem__.return_value.call_args.kwargs
        self.assertTrue(arguments["CONTIGUOUS"])
        self.assertEqual(arguments["STRIDES"], (129, 1))
        self.assertEqual(arguments["INPUT_SPAN"], arguments["BLOCK_COLS"])
        scalar = tensor((), "f32")
        quantize(x, scalar, scalar)
        arguments = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual((arguments["ROW_COUNT"], arguments["COLUMN_COUNT"], arguments["PER_TENSOR"]), (1, 102, True))
        kernel.reset_mock()
        resources.reset_mock()
        self.assertIs(quantize(tensor((0, 17)), reciprocal, offset), output)
        self.assertIs(quantize(tensor((2, 0)), tensor((0,)), tensor((0,))), output)
        kernel.__getitem__.assert_not_called()
        resources.assert_not_called()
        for bad in (tensor((18,)), tensor((17,), "f32"), tensor((17,), npu_format=29)):
            with self.assertRaises(ValueError):
                quantize(x, bad, bad)

    def test_layout_copy_uses_actual_shape_strides_dtype_and_resource_budget(self):
        kernel = MagicMock()
        self.namespace.update(
            triton=SimpleNamespace(
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: (n + d - 1) // d
            ),
            _quantize_layout_copy_kernel=kernel,
        )
        for (shape, axis), element_bytes, cores, ub in itertools.product(
            (((13, 17), 0), ((128, 8192), 0), ((3, 5, 17), 0), ((3, 5, 17), 1), ((2, 4718593), 0)),
            (2, 4),
            (7, 48),
            (8192, 196608),
        ):
            dense = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
            strides, stride = [0] * len(shape), 1
            for dimension in reversed([i for i in range(len(shape)) if i != axis] + [axis]):
                strides[dimension], stride = stride, stride * shape[dimension]
            strides = tuple(strides)
            x = SimpleNamespace(
                shape=shape,
                device=object(),
                dtype=object(),
                numel=lambda shape=shape: math.prod(shape),
                element_size=lambda element_bytes=element_bytes: element_bytes,
                stride=lambda strides=strides: strides,
            )
            out = SimpleNamespace(stride=lambda dense=dense: dense)
            x.view = lambda _, x=x: x
            out.view = lambda _, out=out: out
            self.namespace["torch"] = SimpleNamespace(
                empty=lambda *a, out=out, **k: out, int16=object(), int32=object()
            )
            resources = SimpleNamespace(cache_line_bytes=512, ub_bytes=ub, vector_cores=cores)
            self.assertIs(self.namespace["_quantize_contiguous_copy"](x, axis, resources), out)
            kwargs = kernel.__getitem__.return_value.call_args.kwargs
            a, b = kwargs["INNER_BLOCK"], kwargs["OUTER_BLOCK"]
            self.assertEqual(kwargs["STRIDES"], strides)
            self.assertEqual(kwargs["DENSE_STRIDES"], dense)
            self.assertEqual(kwargs["FULL_TILE"], shape[axis] % a == 0 and shape[-1] % b == 0)
            self.assertLessEqual(a * b * (2 * element_bytes + 2 * 8 + 1), ub - 2048)
            alignment = 32 // element_bytes
            padded_a, padded_b = (
                (a + alignment - 1) // alignment * alignment,
                (b + alignment - 1) // alignment * alignment,
            )
            self.assertLessEqual(2 * padded_a * padded_b * element_bytes + (a + b) * 17, ub - 2048)
            gx, gy, gz = kernel.__getitem__.call_args.args[0]
            self.assertEqual(gz, math.prod(shape) // shape[axis] // shape[-1])
            self.assertGreaterEqual(gx * a, shape[axis])
            self.assertGreaterEqual(gy * b, shape[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args, remaining = parser.parse_known_args()
    if args.source is not None:
        SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
