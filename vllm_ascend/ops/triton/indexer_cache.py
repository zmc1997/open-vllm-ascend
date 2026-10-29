# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent, strided row-scatter writes for indexer caches.

Inputs follow the caller's 2D scatter contract; no model or batch-size policy.
Repeated valid slots keep the last input row. CANN agreement outside the tested
slot domain (-1 or valid rows) still requires an oracle on the target device.
"""

from functools import lru_cache

import torch
import torch_npu
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.kernel_resources import DMA_ALIGNMENT_BYTES, get_device_resources
from vllm_ascend.ops.triton.triton_utils import get_element


@lru_cache(maxsize=None)  # noqa: UP033 -- preserve the original runtime AST.
def _scatter_tiling(rows, width, capacity, element_bytes, index_bytes, value_column_stride, vector_cores, ub_bytes):
    """Budget the concurrently live index and copy vectors; no model-shape policy."""
    index_block = 1 << (rows - 1).bit_length()
    copy_block = 1 << (width - 1).bit_length()

    def buffer_bytes(lanes, element_size):
        return ((lanes * element_size + DMA_ALIGNMENT_BYTES - 1) // DMA_ALIGNMENT_BYTES) * DMA_ALIGNMENT_BYTES

    def index_bytes_live(lanes):
        # Loaded indices, int64 slot/address vectors, and the tail mask.
        return sum(buffer_bytes(lanes, size) for size in (index_bytes, 8, 8, 1))

    def copy_bytes_live(lanes):
        if value_column_stride == 1:
            # Raw payload, int64 source/destination offsets, and the tail mask.
            return sum(buffer_bytes(lanes, size) for size in (element_bytes, 8, 8, 1))
        # The normalization kernel reads one physical span, then gathers the
        # logical words into a contiguous output. Ascend's byte gather uses
        # FP16 internally; budget both raw and widened carriers plus addresses.
        span = 1 << (((lanes - 1) * value_column_stride + 1) - 1).bit_length()
        words = 2 if element_bytes == 8 else 1
        word_bytes = element_bytes // words
        carrier_bytes = max(word_bytes, 2)
        return sum(buffer_bytes(span * words, size) for size in (word_bytes, carrier_bytes, 8, 1)) + sum(
            buffer_bytes(lanes * words, size) for size in (word_bytes, carrier_bytes, 8, 4, 1)
        )

    while index_bytes_live(index_block) + copy_bytes_live(copy_block) > ub_bytes:
        if index_block > 1 and (copy_block == 1 or index_bytes_live(index_block) >= copy_bytes_live(copy_block)):
            index_block //= 2
        elif copy_block > 1:
            copy_block //= 2
        else:
            raise RuntimeError("Device UB cannot hold scatter index and copy buffers")
    return min(rows, capacity, vector_cores), index_block, copy_block


@lru_cache(maxsize=None)  # noqa: UP033 -- preserve the original runtime AST.
def _transpose_tiling(rows, width, element_bytes, vector_cores, ub_bytes):
    """Balance physical DMA work across cores, including both transpose layouts."""
    alignment = DMA_ALIGNMENT_BYTES // element_bytes
    ceildiv = lambda a, b: (a + b - 1) // b
    best = None
    for row_power in range((rows - 1).bit_length() + 1):
        row_block = 1 << row_power
        aligned_rows = ceildiv(row_block, alignment) * alignment
        for column_power in range((width - 1).bit_length() + 1):
            column_block = 1 << column_power
            aligned_columns = ceildiv(column_block, alignment) * alignment
            # Source and transposed UB layouts need their own DMA alignment;
            # include the axis address vectors and masks while both are live.
            working = 2 * aligned_rows * aligned_columns * element_bytes + (row_block + column_block) * (2 * 8 + 1)
            if working > ub_bytes:
                continue
            tasks = ceildiv(rows, row_block) * ceildiv(width, column_block)
            dma_bytes = (column_block * aligned_rows + row_block * aligned_columns) * element_bytes
            cost = (ceildiv(tasks, vector_cores) * dma_bytes, tasks)
            if best is None or cost < best[0]:
                best = cost, row_block, column_block
    if best is None:
        raise RuntimeError("Device UB cannot hold a DMA-aligned transpose tile")
    return best[1], best[2]


@triton.jit
def _indexer_cache_transpose_kernel(
    values,
    output,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    VALUE_COLUMN_STRIDE: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
    FULL: tl.constexpr,
):
    # Read the source's physical unit-stride row axis, then write contiguous
    # output columns. Raw integer carriers never convert floating payloads.
    if ELEMENT_BYTES == 1:
        RAW_TYPE: tl.constexpr = tl.int8
    elif ELEMENT_BYTES == 2:
        RAW_TYPE: tl.constexpr = tl.int16
    else:
        RAW_TYPE: tl.constexpr = tl.int32
    source = values.to(tl.pointer_type(RAW_TYPE))
    target = output.to(tl.pointer_type(RAW_TYPE))
    first_row = tl.program_id(0).to(tl.int64) * ROW_BLOCK
    first_column = tl.program_id(1).to(tl.int64) * COLUMN_BLOCK
    row = tl.arange(0, ROW_BLOCK)
    column = tl.arange(0, COLUMN_BLOCK)
    source_origin = source + first_column * VALUE_COLUMN_STRIDE + first_row
    target_origin = target + first_row * WIDTH + first_column
    # Keep only the scalar GM origins wide. Promote a local axis only when
    # its complete tile offset cannot fit in a signed 32-bit integer.
    if (COLUMN_BLOCK - 1) * VALUE_COLUMN_STRIDE + ROW_BLOCK > (1 << 31) - 1:
        source_column = column.to(tl.int64)
    else:
        source_column = column
    if (ROW_BLOCK - 1) * WIDTH + COLUMN_BLOCK > (1 << 31) - 1:
        target_row = row.to(tl.int64)
    else:
        target_row = row
    source_tile = source_origin + source_column[:, None] * VALUE_COLUMN_STRIDE + row[None, :]
    target_tile = target_origin + target_row[:, None] * WIDTH + column[None, :]
    if FULL:
        value = tl.load(source_tile)
        tl.store(target_tile, tl.trans(value))
    else:
        row_valid = row < ROWS - first_row
        column_valid = column < WIDTH - first_column
        value = tl.load(source_tile, mask=column_valid[:, None] & row_valid[None, :], other=0)
        # Build the target mask directly. Transposing a boolean mask causes
        # read/modify/write lowering on the installed Ascend compiler.
        tl.store(target_tile, tl.trans(value), mask=row_valid[:, None] & column_valid[None, :])


@triton.jit
def _indexer_cache_copy_kernel(
    values,
    output,
    WIDTH: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    VALUE_COLUMN_STRIDE: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
):
    """Normalize raw updates before scatter, matching CANN's Contiguous stage.

    A separate contiguous destination keeps the output scatter stride from
    propagating into the gather's UB layout on the installed backend.
    """
    row = tl.program_id(0).to(tl.int64)
    begin = tl.program_id(1).to(tl.int64) * COPY_BLOCK
    if ELEMENT_BYTES == 1:
        WORD_TYPE: tl.constexpr = tl.uint8
        GATHER_TYPE: tl.constexpr = tl.int8
    elif ELEMENT_BYTES == 2:
        WORD_TYPE: tl.constexpr = tl.uint16
        GATHER_TYPE: tl.constexpr = tl.float16
    else:
        WORD_TYPE: tl.constexpr = tl.uint32
        GATHER_TYPE: tl.constexpr = tl.float32
    WORDS: tl.constexpr = 2 if ELEMENT_BYTES == 8 else 1
    source_words = values.to(tl.pointer_type(WORD_TYPE))
    output_words = output.to(tl.pointer_type(WORD_TYPE))
    lanes = tl.arange(0, COPY_BLOCK * WORDS)
    physical = tl.arange(0, SPAN * WORDS)
    valid_span = (tl.minimum(COPY_BLOCK, WIDTH - begin) - 1) * VALUE_COLUMN_STRIDE + 1
    payload = tl.load(
        source_words + (row * VALUE_ROW_STRIDE + begin * VALUE_COLUMN_STRIDE) * WORDS + physical,
        mask=physical < valid_span * WORDS,
        other=0,
    ).to(GATHER_TYPE, bitcast=True)
    # Supported bitcast carriers preserve payload bits. Gather both halves of
    # 64-bit elements together, so the destination words remain contiguous.
    value = tl.gather(payload, (lanes // WORDS) * VALUE_COLUMN_STRIDE * WORDS + lanes % WORDS, axis=0)
    tl.store(
        output_words + (row * WIDTH + begin) * WORDS + lanes,
        value.to(WORD_TYPE, bitcast=True),
        mask=begin * WORDS + lanes < WIDTH * WORDS,
    )


@triton.jit
def _indexer_cache_scatter_kernel(
    cache,
    slots,
    values,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_COLUMN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    VALUE_COLUMN_STRIDE: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    OWNERS: tl.constexpr,
    INDEX_BLOCK: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
):
    owner = tl.program_id(0)
    rows_per_owner: tl.constexpr = CACHE_ROWS // OWNERS
    extra_rows: tl.constexpr = CACHE_ROWS % OWNERS
    first = owner.to(tl.int64) * rows_per_owner + tl.minimum(owner, extra_rows)
    end = first + rows_per_owner + (owner < extra_rows)
    tl.static_assert((ELEMENT_BYTES == 1) | (ELEMENT_BYTES == 2) | (ELEMENT_BYTES == 4) | (ELEMENT_BYTES == 8))
    if ELEMENT_BYTES == 1:
        raw_type: tl.constexpr = tl.uint8
    elif ELEMENT_BYTES == 2:
        raw_type: tl.constexpr = tl.uint16
    elif ELEMENT_BYTES == 4:
        raw_type: tl.constexpr = tl.uint32
    else:
        raw_type: tl.constexpr = tl.uint64
    raw_cache = cache.to(tl.pointer_type(raw_type))
    raw_values = values.to(tl.pointer_type(raw_type))
    # CANN NoSort organization: disjoint output owners scan input rows in order.
    # No N-by-N duplicate search or workspace. Sparse nearby slots may serialize
    # on one owner; this choice still needs comparison with CANN's sorted path.
    for index_begin in tl.range(0, N, INDEX_BLOCK):
        input_rows = index_begin + tl.arange(0, INDEX_BLOCK)
        slot_block = tl.load(slots + input_rows.to(tl.int64) * SLOT_STRIDE, mask=input_rows < N, other=-1).to(tl.int64)
        for index in tl.range(0, tl.minimum(INDEX_BLOCK, N - index_begin)):
            row = index_begin + index
            slot = get_element(slot_block, (index,))
            # One owner performs every update to a slot, in input order. Keep
            # loop-carried store dependencies (multibuffer=False at launch).
            if (slot >= first) & (slot < end):
                for begin in tl.range(0, WIDTH, COPY_BLOCK):
                    columns = (begin + tl.arange(0, COPY_BLOCK)).to(tl.int64)
                    value = tl.load(
                        raw_values + row.to(tl.int64) * VALUE_ROW_STRIDE + columns * VALUE_COLUMN_STRIDE,
                        mask=columns < WIDTH,
                        other=0,
                    )
                    tl.store(
                        raw_cache + slot * CACHE_ROW_STRIDE + columns * CACHE_COLUMN_STRIDE,
                        value,
                        mask=columns < WIDTH,
                    )


def _scatter_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if (
        cache.ndim != 2
        or values.ndim != 2
        or cache.shape[1] != values.shape[1]
        or cache.dtype != values.dtype
        or cache.device.type != "npu"
        or values.device != cache.device
        or slots.device != cache.device
        or slots.dtype not in (torch.int32, torch.int64)
        or slots.shape not in ((values.shape[0],), (values.shape[0], 1))
    ):
        raise ValueError("Row scatter requires same-width/dtype 2D cache and values, and N integer slots on one NPU")
    if any(torch_npu.get_npu_format(tensor) != 2 for tensor in (cache, slots, values)):
        raise RuntimeError("ScatterNdUpdate requires ND tensors; non-contiguous ND strides are supported")
    rows, width = values.shape
    if rows and width and cache.shape[0]:
        resources = get_device_resources(cache.device.index)
        # A single column has no strided vector gather to normalize.
        if width > 1 and values.stride(1) != 1:
            contiguous = torch.empty(values.shape, dtype=values.dtype, device=values.device)
            # The transpose uses 8/16/32-bit carriers. Keep the validated
            # two-word span copy for 64-bit data, preserving all payload bits.
            if values.stride(0) == 1 and values.stride(1) > 1 and values.element_size() <= 4:
                row_block, column_block = _transpose_tiling(
                    rows,
                    width,
                    values.element_size(),
                    resources.vector_cores,
                    resources.ub_bytes,
                )
                _indexer_cache_transpose_kernel[(triton.cdiv(rows, row_block), triton.cdiv(width, column_block))](
                    values,
                    contiguous,
                    ROWS=rows,
                    WIDTH=width,
                    VALUE_COLUMN_STRIDE=values.stride(1),
                    ELEMENT_BYTES=values.element_size(),
                    ROW_BLOCK=row_block,
                    COLUMN_BLOCK=column_block,
                    FULL=rows % row_block == 0 and width % column_block == 0,
                    multibuffer=False,
                )
            else:
                _, _, copy_block = _scatter_tiling(
                    rows,
                    width,
                    cache.shape[0],
                    cache.element_size(),
                    slots.element_size(),
                    values.stride(1),
                    resources.vector_cores,
                    resources.ub_bytes,
                )
                _indexer_cache_copy_kernel[(rows, triton.cdiv(width, copy_block))](
                    values,
                    contiguous,
                    WIDTH=width,
                    VALUE_ROW_STRIDE=values.stride(0),
                    VALUE_COLUMN_STRIDE=values.stride(1),
                    ELEMENT_BYTES=values.element_size(),
                    COPY_BLOCK=copy_block,
                    SPAN=triton.next_power_of_2((copy_block - 1) * values.stride(1) + 1),
                    multibuffer=False,
                )
            values = contiguous
        owners, index_block, copy_block = _scatter_tiling(
            rows,
            width,
            cache.shape[0],
            cache.element_size(),
            slots.element_size(),
            values.stride(1),
            resources.vector_cores,
            resources.ub_bytes,
        )
        _indexer_cache_scatter_kernel[(owners,)](
            cache,
            slots,
            values,
            N=rows,
            WIDTH=width,
            CACHE_ROWS=cache.shape[0],
            CACHE_ROW_STRIDE=cache.stride(0),
            CACHE_COLUMN_STRIDE=cache.stride(1),
            SLOT_STRIDE=slots.stride(0),
            VALUE_ROW_STRIDE=values.stride(0),
            VALUE_COLUMN_STRIDE=values.stride(1),
            ELEMENT_BYTES=cache.element_size(),
            OWNERS=owners,
            INDEX_BLOCK=index_block,
            COPY_BLOCK=copy_block,
            multibuffer=False,
        )
    return cache


def scatter_k_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Internal ND row bit-copy; preserve cache alias and actual tensor strides."""
    return _scatter_cache(cache, slots, values)


def scatter_scale_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Independent scale row bit-copy, with no fixed width or floating dtype."""
    return _scatter_cache(cache, slots, values)
