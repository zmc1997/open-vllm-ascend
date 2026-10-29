# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ND QINT8 quantization with independent core partition and UB tiles.

Read the original stored reciprocal/offset, never recompute the reciprocal.
Per-channel parameters retain their actual dtype, address and stride. The
existing FP32 scalar interface remains supported. Candidate layouts and dtypes
must pass the strict CANN oracle on the actual NPU before deployment.
"""

import torch
import torch_npu
import triton.language.extra.cann.extension as al
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.kernel_resources import DMA_ALIGNMENT_BYTES, get_device_resources


def _cann_ub_bytes(rows: int, columns: int, element_bytes: int, parameter_bytes: int, per_tensor: bool) -> int:
    """CANN v9.1's explicit buffers, NOT a Triton allocation measurement.

    FP16/BF16 retain a batch of raw input/output and one FP32 work row.
    FP32 modifies its input batch in place. Channel parameters remain live
    across all row batches; only low-precision parameters need cast buffers.
    The Triton lowering must independently demonstrate this buffer reuse.
    """

    def aligned(size):
        return (size + DMA_ALIGNMENT_BYTES - 1) // DMA_ALIGNMENT_BYTES * DMA_ALIGNMENT_BYTES

    data = aligned(rows * columns * element_bytes) + aligned(rows * columns)
    if element_bytes < 4:
        data += aligned(columns * 4)
    if per_tensor:
        # FP16/BF16 use a DMA block and GetValue. FP32 reads a GM scalar.
        parameters = 2 * aligned(parameter_bytes) if parameter_bytes < 4 else 0
    else:
        parameters = 2 * aligned(columns * parameter_bytes)
        if parameter_bytes < 4:
            parameters += 2 * aligned(columns * 4)
    return data + parameters


def _cann_ub_factors(
    row_factor: int,
    column_factor: int,
    element_bytes: int,
    per_tensor: bool,
    double_cut: bool,
    ub_bytes: int,
    cache_line_bytes: int,
) -> tuple[int, int]:
    """Translate CalcMaxBaseLen/CalcMaxN/CalcUBFactor, including its reserve.

    These factors describe the original implementation. They are separate
    from Triton's power-of-two carriers and compiler-owned temporary storage.
    """
    reserve_bytes = 2048  # AscendQuantV2::reserveUb_, not a tuned tile size.
    available = ub_bytes - reserve_bytes
    cast_bytes = 4 if element_bytes < 4 else 0
    input_count = 1 if per_tensor else 3  # x, reciprocal, offset.
    bytes_per_column = input_count * (element_bytes + cast_bytes) + 1
    line_elements = cache_line_bytes // element_bytes
    max_columns = available // bytes_per_column // line_elements * line_elements
    if max_columns < line_elements:
        raise RuntimeError("UB cannot hold one CANN-aligned static quantization tile")
    block_columns = (column_factor + line_elements - 1) // line_elements * line_elements
    if not double_cut and block_columns <= max_columns // 2:
        parameter_count = 0 if per_tensor else 2
        remaining = available - block_columns * (parameter_count * (element_bytes + cast_bytes) + cast_bytes)
        max_rows = max(1, remaining // (element_bytes + 1) // block_columns)
        return min(row_factor, max_rows), block_columns
    return 1, min(block_columns, max_columns)


def _quantize_ub_bytes(
    rows: int,
    columns: int,
    element_bytes: int,
    parameter_bytes: int,
    contiguous: bool,
    parameter_contiguous: bool,
    per_tensor: bool,
) -> int:
    """Conservative carrier working set; verify against each compiled kernel.

    CANN's buffer formula is only the starting point. Triton additionally
    needs row arithmetic/select temporaries, addressing and masks. A multirow
    result uses a FP16 carrier before the final INT8 cast: A3's INT8
    insert_slice otherwise recasts the whole batch twice for every row.
    Strided DMA stages one 32B block per lane in the observed A3 lowering.
    This source bound is not a compiler-reported allocation or proof of reuse.
    """

    def aligned(size):
        return (size + DMA_ALIGNMENT_BYTES - 1) // DMA_ALIGNMENT_BYTES * DMA_ALIGNMENT_BYTES

    data = _cann_ub_bytes(rows, columns, element_bytes, parameter_bytes, per_tensor)
    if rows == 1 and contiguous and (parameter_contiguous or per_tensor):
        # The affine row path lowers addresses/tail lengths to scalars. Its
        # previous physical IR reused the FP32 work row through arithmetic
        # and casts. Keep its extra row/predicate allowance as a conservative
        # upper bound after removing the explicit NaN select; do not shrink
        # the budget without checking the new compiler's buffer allocation.
        return data + aligned(columns * 4) + aligned((columns + 7) // 8)
    # Keep up to three FP32 row buffers; only low-precision CANN input has
    # already contributed one. The actual reuse is checked in physical IR.
    data += (2 if element_bytes < 4 else 3) * aligned(columns * 4)
    data += 2 * aligned(columns * 8) + 3 * aligned(columns) + aligned(rows * 8)
    if rows > 1:
        data += aligned(rows * columns * 2)
    if not contiguous:
        data += 3 * aligned(columns * 8) + aligned(rows * columns * DMA_ALIGNMENT_BYTES)
    if not per_tensor and not parameter_contiguous:
        data += 2 * aligned(columns * (8 + DMA_ALIGNMENT_BYTES))
    return data


def _quantize_tiling(
    rows: int,
    width: int,
    element_bytes: int,
    parameter_bytes: int,
    contiguous: bool,
    parameter_contiguous: bool,
    per_tensor: bool,
    vector_cores: int,
    ub_bytes: int,
    cache_line_bytes: int,
) -> dict[str, int]:
    """Resource-sized row tasks, then CANN partitioning and UB factors.

    Source: CANN v9.1 AscendQuantV2 CalcTiling/GetCoreNum/CalcBlockFactor.
    Scalar parameters use logical [1, numel]; channel parameters use
    [prod(shape[:-1]), shape[-1]]. For contiguous channel input with fewer
    rows than cores, avoid splitting each UB-sized row piece into many small
    programs. This is a Triton scheduling choice, not CANN's hardware-pool
    policy. At least one row per hardware core, strided layouts and scalar
    parameters retain the original pool and its established batch behavior.
    UB factors follow CalcUBFactor/CalcUBFactorDoubleCut. Power-of-two
    carriers are accounted separately, without changing core ownership.
    Original buffer sizes do not prove the compiler reuses Triton buffers.
    """
    core_pool = vector_cores
    if contiguous and parameter_contiguous and not per_tensor and rows < vector_cores:
        _, row_columns = _cann_ub_factors(1, width, element_bytes, False, False, ub_bytes, cache_line_bytes)
        row_columns = 1 << (row_columns.bit_length() - 1)
        while _quantize_ub_bytes(1, row_columns, element_bytes, parameter_bytes, True, True, False) > ub_bytes - 2048:
            if row_columns == DMA_ALIGNMENT_BYTES:
                raise RuntimeError("UB cannot hold one aligned static quantization tile")
            row_columns //= 2
        row_pieces = (width + row_columns - 1) // row_columns
        core_pool = min(vector_cores, rows * row_pieces)

    def active(parts, pool):
        work = (parts + pool - 1) // pool
        return (parts + work - 1) // work

    line_elements = cache_line_bytes // element_bytes
    column_units = (width + line_elements - 1) // line_elements
    # CANN GetNewShape0 binds short output rows on non-V220 architectures.
    row_bind = (
        (DMA_ALIGNMENT_BYTES + width - 1) // width if cache_line_bytes != 512 and width < DMA_ALIGNMENT_BYTES else 1
    )
    partition_rows = (rows + row_bind - 1) // row_bind
    row_cores, column_cores = active(partition_rows, core_pool), active(column_units, core_pool)
    if row_cores >= column_cores:
        cores, row_factor = row_cores, (partition_rows + row_cores - 1) // row_cores * row_bind
        column_factor, column_cores = width, 1
    else:
        cores, row_factor = column_cores, rows
        column_factor = (column_units + column_cores - 1) // column_cores * line_elements
    # The width guard is CANN's g_BlockSize (32), not a preferred tile size.
    double_cut = False
    if cores < core_pool and rows > 1 and width > DMA_ALIGNMENT_BYTES and core_pool // rows:
        double_columns = active(column_units, core_pool // rows)
        if rows * double_columns > cores:
            cores, row_factor, column_cores = rows * double_columns, 1, double_columns
            column_factor = (column_units + column_cores - 1) // column_cores * line_elements
            double_cut = True

    base_rows, base_columns = _cann_ub_factors(
        row_factor, column_factor, element_bytes, per_tensor, double_cut, ub_bytes, cache_line_bytes
    )
    # A smaller power-of-two column carrier never exceeds the original
    # column buffer. Rows may have a masked tail, so account their carrier.
    block_columns = 1 << (base_columns.bit_length() - 1)
    block_rows = 1 << (base_rows - 1).bit_length()
    while (
        _quantize_ub_bytes(
            block_rows, block_columns, element_bytes, parameter_bytes, contiguous, parameter_contiguous, per_tensor
        )
        > ub_bytes - 2048
    ):
        if block_rows > 1:
            block_rows //= 2
            base_rows = min(base_rows, block_rows)
        elif block_columns > DMA_ALIGNMENT_BYTES:
            block_columns //= 2
        else:
            raise RuntimeError("UB cannot hold one aligned static quantization tile")
    return {
        "NUM_CORES": cores,
        "ROW_FACTOR": row_factor,
        "COLUMN_FACTOR": column_factor,
        "COLUMN_CORES": column_cores,
        "BASE_N": base_rows,
        "BLOCK_ROWS": block_rows,
        "BLOCK_COLS": block_columns,
    }


@triton.jit
def _quantize_row(x, reciprocal, offset):
    """One CANN Compute row; preserve the established FP32 rounding contract."""
    product = x.to(tl.float32) * reciprocal
    affine = product + offset
    # Preserve NaNs through rounding. The final saturating device cast has
    # CANN's NaN-to-zero behavior; gate it with all BF16/FP16 bit patterns.
    bounded = tl.minimum(
        tl.maximum(affine, -128.0, propagate_nan=tl.PropagateNan.ALL),
        127.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    RNE_MAGIC: tl.constexpr = 12582912.0
    rounded = (bounded + RNE_MAGIC) - RNE_MAGIC
    # These finite, bounded integral FP32 values are exactly representable in FP16.
    # CANN uses an INT32 carrier between RINT and FP16, but this backend
    # lowers that carrier through INT32 -> FP32 -> FP16. Keep the already
    # rounded value instead of adding a redundant integer round trip.
    return rounded.to(tl.float16)


@triton.jit
def _quantize_layout_copy_kernel(
    source,
    output,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DENSE_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUTER: tl.constexpr,
    INNER_BLOCK: tl.constexpr,
    OUTER_BLOCK: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    """Bit-copy an ND view, reading along its physically contiguous axis."""
    inner_start = tl.program_id(0).to(tl.int64) * INNER_BLOCK
    outer_start = tl.program_id(1).to(tl.int64) * OUTER_BLOCK
    inner = tl.arange(0, INNER_BLOCK)
    outer = tl.arange(0, OUTER_BLOCK)
    source_base = tl.full((), 0, tl.int64)
    output_base = tl.full((), 0, tl.int64)
    remaining = tl.program_id(2).to(tl.int64)
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        if axis != AXIS:
            coordinate = remaining % SHAPE.value[axis]
            remaining = remaining // SHAPE.value[axis]
            source_base += coordinate * STRIDES.value[axis]
            output_base += coordinate * DENSE_STRIDES.value[axis]
    source_origin = source + source_base + outer_start * STRIDES.value[-1] + inner_start
    output_origin = output + output_base + inner_start * DENSE_STRIDES.value[AXIS] + outer_start
    # Keep the GM tile origin in INT64. Local offsets fit INT32 unless the
    # actual stride requires a wider address; avoid vector INT64 conversion.
    MAX_INT32_OFFSET: tl.constexpr = (1 << 31) - 1
    if (OUTER_BLOCK - 1) * STRIDES.value[-1] + INNER_BLOCK > MAX_INT32_OFFSET:
        source_outer = outer.to(tl.int64)
    else:
        source_outer = outer
    if (INNER_BLOCK - 1) * DENSE_STRIDES.value[AXIS] + OUTER_BLOCK > MAX_INT32_OFFSET:
        output_inner = inner.to(tl.int64)
    else:
        output_inner = inner
    source_tile = source_origin + source_outer[:, None] * STRIDES.value[-1] + inner[None, :]
    output_tile = output_origin + output_inner[:, None] * DENSE_STRIDES.value[AXIS] + outer[None, :]
    if FULL_TILE:
        value = tl.load(source_tile)
        tl.store(output_tile, tl.trans(value))
    else:
        valid_inner = inner < SHAPE.value[AXIS] - inner_start
        valid_outer = outer < OUTER - outer_start
        value = tl.load(source_tile, valid_outer[:, None] & valid_inner[None, :], other=0)
        # Rebuild the rectangular mask in output order. Transposing a boolean
        # mask triggered a read/modify/write lowering on the installed backend.
        tl.store(output_tile, tl.trans(value), valid_inner[:, None] & valid_outer[None, :])


def _quantize_contiguous_copy(x, axis, resources):
    """Normalize costly transposed views with Triton; include this call in timing."""
    width = x.shape[axis]
    outer = x.shape[-1]
    batches = x.numel() // width // outer
    inner_block = min(triton.next_power_of_2(width), resources.cache_line_bytes // x.element_size())
    # Two payload carriers, two INT64 address vectors, and the predicate.
    capacity = (resources.ub_bytes - 2048) // (2 * x.element_size() + 2 * 8 + 1)
    per_core = triton.cdiv(x.numel(), resources.vector_cores)
    outer_block = min(
        triton.next_power_of_2(outer),
        triton.next_power_of_2(triton.cdiv(per_core, inner_block)),
        1 << (max(1, capacity // inner_block).bit_length() - 1),
    )
    alignment = DMA_ALIGNMENT_BYTES // x.element_size()
    while (
        2
        * triton.cdiv(inner_block, alignment)
        * alignment
        * triton.cdiv(outer_block, alignment)
        * alignment
        * x.element_size()
        + (inner_block + outer_block) * (2 * 8 + 1)
        > resources.ub_bytes - 2048
    ):
        # Transpose carriers pad BOTH axes to a DMA block, including narrow
        # rows. Logical element count alone misses this physical allocation.
        if outer_block > 1:
            outer_block //= 2
        elif inner_block > 1:
            inner_block //= 2
        else:
            raise RuntimeError("UB cannot hold one DMA-aligned transpose tile")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    raw_dtype = torch.int16 if x.element_size() == 2 else torch.int32
    _quantize_layout_copy_kernel[(triton.cdiv(width, inner_block), triton.cdiv(outer, outer_block), batches)](
        x.view(raw_dtype),
        result.view(raw_dtype),
        SHAPE=tuple(x.shape),
        STRIDES=tuple(x.stride()),
        DENSE_STRIDES=tuple(result.stride()),
        AXIS=axis,
        OUTER=outer,
        INNER_BLOCK=inner_block,
        OUTER_BLOCK=outer_block,
        FULL_TILE=width % inner_block == 0 and outer % outer_block == 0,
        multibuffer=False,
    )
    return result


@triton.jit
def _load_quantize_columns(
    ptr,
    start,
    end,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    DMA_BYTES: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    FULL: tl.constexpr,
    SPAN: tl.constexpr,
):
    """Coalesce short-stride reads before selecting the logical columns.

    The A3 strided DMA lowering otherwise reserves a 32-byte block for each
    element. Read the physical span when it is smaller than that staging
    space; selection stays in UB. No address outside the valid row is read.
    """
    lanes = tl.arange(0, BLOCK)
    if STRIDE > 1 and SPAN * ELEMENT_BYTES < BLOCK * DMA_BYTES:
        physical = tl.arange(0, SPAN)
        if FULL:
            span_size = (BLOCK - 1) * STRIDE + 1
        else:
            span_size = (end - start - 1) * STRIDE + 1
        values = tl.load(ptr + start * STRIDE + physical, physical < span_size, other=0)
        return tl.gather(values, lanes * STRIDE, axis=0)
    else:
        if FULL:
            return tl.load(ptr + (start + lanes) * STRIDE)
        else:
            return tl.load(ptr + (start + lanes) * STRIDE, start + lanes < end, other=0)


@triton.jit
def _static_quantize_kernel(
    x_ptr,
    reciprocal_ptr,
    offset_ptr,
    output_ptr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    ROW_COUNT: tl.constexpr,
    COLUMN_COUNT: tl.constexpr,
    PER_TENSOR: tl.constexpr,
    RECIPROCAL_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
    INPUT_SPAN: tl.constexpr,
    RECIPROCAL_SPAN: tl.constexpr,
    OFFSET_SPAN: tl.constexpr,
    ROW_FACTOR: tl.constexpr,
    COLUMN_FACTOR: tl.constexpr,
    COLUMN_CORES: tl.constexpr,
    BASE_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    PARAMETER_BLOCK: tl.constexpr,
    DMA_BYTES: tl.constexpr,
    INPUT_BYTES: tl.constexpr,
    PARAMETER_BYTES: tl.constexpr,
):
    core = tl.program_id(0).to(tl.int64)
    row_start = (core // COLUMN_CORES) * ROW_FACTOR
    column_start = (core % COLUMN_CORES) * COLUMN_FACTOR
    row_end = tl.minimum(row_start + ROW_FACTOR, ROW_COUNT)
    column_end = tl.minimum(column_start + COLUMN_FACTOR, COLUMN_COUNT)
    # Like CANN's full-length / tail loops, prove the full tile from the
    # partition dimensions. Avoid dynamic padding and tail bookkeeping when
    # every column tile is complete; arbitrary tails retain masked accesses.
    FULL_COLUMNS: tl.constexpr = COLUMN_COUNT % COLUMN_FACTOR == 0 and COLUMN_FACTOR % BLOCK_COLS == 0
    if PER_TENSOR:
        if reciprocal_ptr.dtype.element_ty == tl.float32:
            # Original PerTensorFP32 uses GM GetValue. Retain Tensor updates.
            reciprocal = tl.load(reciprocal_ptr)
            offset = tl.load(offset_ptr)
        else:
            # PerTensorFP16/BF16 stages a DMA block, then reads its first value.
            # Masked lanes never read beyond a one-element parameter Tensor.
            lanes = tl.arange(0, PARAMETER_BLOCK)
            raw_reciprocal = tl.load(reciprocal_ptr + lanes, lanes == 0, other=0)
            raw_offset = tl.load(offset_ptr + lanes, lanes == 0, other=0)
            reciprocal = al.get_element(raw_reciprocal, (0,)).to(tl.float32)
            offset = al.get_element(raw_offset, (0,)).to(tl.float32)
    for column_tile in range(0, column_end - column_start, BLOCK_COLS):
        columns = column_start + column_tile + tl.arange(0, BLOCK_COLS)
        if FULL_COLUMNS:
            column_mask = tl.full((BLOCK_COLS,), True, tl.int1)
        else:
            column_mask = columns < column_end
        if not PER_TENSOR:
            # Scalar tile origin plus consecutive lanes gives a contiguous
            # slice for stride=1. Strided parameter views keep their semantics.
            reciprocal = _load_quantize_columns(
                reciprocal_ptr,
                column_start + column_tile,
                column_end,
                RECIPROCAL_STRIDE,
                BLOCK_COLS,
                DMA_BYTES,
                PARAMETER_BYTES,
                FULL_COLUMNS,
                RECIPROCAL_SPAN,
            ).to(tl.float32)
            offset = _load_quantize_columns(
                offset_ptr,
                column_start + column_tile,
                column_end,
                OFFSET_STRIDE,
                BLOCK_COLS,
                DMA_BYTES,
                PARAMETER_BYTES,
                FULL_COLUMNS,
                OFFSET_SPAN,
            ).to(tl.float32)
        for row_batch in range(row_start, row_end, BASE_N):
            if BASE_N == 1:
                index = row_batch * COLUMN_COUNT + columns
                if CONTIGUOUS:
                    input_index = index
                elif not PER_TENSOR:
                    # Decode the leading dimensions once per logical row;
                    # the column slice retains its affine physical stride.
                    row_offset = tl.full((), 0, tl.int64)
                    remaining = row_batch
                    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
                        row_offset += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                        remaining = remaining // SHAPE.value[axis]
                else:
                    input_index = tl.full((BLOCK_COLS,), 0, tl.int64)
                    remaining = index
                    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                        input_index += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                        remaining = remaining // SHAPE.value[axis]
                if not CONTIGUOUS and not PER_TENSOR:
                    x = _load_quantize_columns(
                        x_ptr + row_offset,
                        column_start + column_tile,
                        column_end,
                        STRIDES.value[-1],
                        BLOCK_COLS,
                        DMA_BYTES,
                        INPUT_BYTES,
                        FULL_COLUMNS,
                        INPUT_SPAN,
                    )
                else:
                    x = tl.load(x_ptr + input_index, mask=column_mask, other=0)
                half = _quantize_row(x, reciprocal, offset)
                tl.store(output_ptr + index, al.cast(half, tl.int8, overflow_mode="saturate"), mask=column_mask)
            else:
                rows = row_batch + tl.arange(0, BLOCK_ROWS)
                batch_end = tl.minimum(row_batch + BASE_N, row_end)
                mask = (rows[:, None] < batch_end) & column_mask[None, :]
                index = rows[:, None] * COLUMN_COUNT + columns[None, :]
                if CONTIGUOUS:
                    batch_x = tl.load(x_ptr + index, mask=mask, other=0)
                # A3 insert_slice on INT8 converts the whole batch twice per
                # row. FP16 preserves every rounded INT8 value exactly and
                # permits one final narrowing cast after all row insertions.
                batch_y = tl.full((BLOCK_ROWS, BLOCK_COLS), 0, tl.float16)
                for row in range(batch_end - row_batch):
                    if CONTIGUOUS:
                        row_x = al.extract_slice(batch_x, (row, 0), (1, BLOCK_COLS), (1, 1))
                        x = tl.reshape(row_x, (BLOCK_COLS,))
                    else:
                        # A3's strided 2D load lowering creates a dynamic
                        # staging slice that fails static buffer-size analysis.
                        # Read each physical row directly; retain the same
                        # parameter reuse and batched contiguous output.
                        row_offset = tl.full((), 0, tl.int64)
                        remaining = row_batch + row
                        for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
                            row_offset += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                            remaining = remaining // SHAPE.value[axis]
                        x = _load_quantize_columns(
                            x_ptr + row_offset,
                            column_start + column_tile,
                            column_end,
                            STRIDES.value[-1],
                            BLOCK_COLS,
                            DMA_BYTES,
                            INPUT_BYTES,
                            FULL_COLUMNS,
                            INPUT_SPAN,
                        )
                    half = _quantize_row(x, reciprocal, offset)
                    batch_y = al.insert_slice(
                        batch_y, tl.reshape(half, (1, BLOCK_COLS)), (row, 0), (1, BLOCK_COLS), (1, 1)
                    )
                flat_y = tl.reshape(batch_y, (BLOCK_ROWS * BLOCK_COLS,))
                result = al.cast(flat_y, tl.int8, overflow_mode="saturate")
                tl.store(output_ptr + index, tl.reshape(result, (BLOCK_ROWS, BLOCK_COLS)), mask=mask)


def quantize(x: torch.Tensor, reciprocal: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Quantize original scalar/channel parameters; return contiguous INT8.

    Accept rank 1..8 strided ND BF16/FP16/FP32 inputs. Parameters are matching
    ND scalars (FP32 or input dtype) or 1D channel vectors in the input dtype.
    Both parameter vectors use their actual strides and remain Tensor inputs
    during graph replay. No host value reads, reciprocal recomputation, or
    persistent parameter copies are performed here.
    """
    if (
        x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or x.layout != torch.strided
        or not 1 <= x.ndim <= 8
    ):
        raise ValueError("Static Triton QINT8 quantization requires a rank 1..8 strided BF16, FP16 or FP32 tensor")
    if x.device.type != "npu" or x.device != reciprocal.device or x.device != offset.device:
        raise ValueError("Static Triton quantization input and parameters must share one NPU device")
    parameter_count = reciprocal.numel()
    per_tensor = parameter_count == 1
    if (
        reciprocal.layout != torch.strided
        or offset.layout != torch.strided
        or reciprocal.shape != offset.shape
        or reciprocal.dtype != offset.dtype
        or reciprocal.ndim not in (0, 1)
        or (per_tensor and reciprocal.dtype not in (x.dtype, torch.float32))
        or (not per_tensor and (reciprocal.ndim != 1 or parameter_count != x.shape[-1] or reciprocal.dtype != x.dtype))
    ):
        raise ValueError("Quantization needs matching scalar parameters or input-dtype last-dimension vectors")
    if any(torch_npu.get_npu_format(tensor) != 2 for tensor in (x, reciprocal, offset)):
        raise ValueError("The original AscendQuantV3 QINT8 interface requires ND storage; NZ is unsupported")
    output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    size = x.numel()
    if size:
        resources = get_device_resources(x.device.index)
        contiguous = x.is_contiguous()
        if not contiguous:
            axis = next((i for i in range(x.ndim) if x.shape[i] > 1 and x.stride(i) == 1), None)
            if (
                axis is not None
                and axis != x.ndim - 1
                and (per_tensor or x.stride(-1) * x.element_size() >= DMA_ALIGNMENT_BYTES)
            ):
                x = _quantize_contiguous_copy(x, axis, resources)
                contiguous = True
        reciprocal_stride = reciprocal.stride()[0] if reciprocal.ndim else 0
        offset_stride = offset.stride()[0] if offset.ndim else 0
        width = size if per_tensor else x.shape[-1]
        rows = 1 if per_tensor else size // width
        plan = _quantize_tiling(
            rows,
            width,
            x.element_size(),
            reciprocal.element_size(),
            contiguous,
            reciprocal_stride == 1 and offset_stride == 1,
            per_tensor,
            resources.vector_cores,
            resources.ub_bytes,
            resources.cache_line_bytes,
        )
        cores = plan.pop("NUM_CORES")
        _static_quantize_kernel[(cores,)](
            x,
            reciprocal,
            offset,
            output,
            SHAPE=tuple(x.shape),
            STRIDES=tuple(x.stride()),
            CONTIGUOUS=contiguous,
            ROW_COUNT=rows,
            COLUMN_COUNT=width,
            PER_TENSOR=per_tensor,
            RECIPROCAL_STRIDE=reciprocal_stride,
            OFFSET_STRIDE=offset_stride,
            # Worker patches replace next_power_of_2 with an int-only helper.
            # Derive the same spans on the host, before entering Triton JIT.
            INPUT_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * x.stride(-1) + 1),
            RECIPROCAL_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * reciprocal_stride + 1),
            OFFSET_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * offset_stride + 1),
            PARAMETER_BLOCK=DMA_ALIGNMENT_BYTES // reciprocal.element_size(),
            DMA_BYTES=DMA_ALIGNMENT_BYTES,
            **plan,
            INPUT_BYTES=x.element_size(),
            PARAMETER_BYTES=reciprocal.element_size(),
            enable_fp_fusion=False,
            multibuffer=False,
            enable_ubuf_saving=True,
        )
    return output
