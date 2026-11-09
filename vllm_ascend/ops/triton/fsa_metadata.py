# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.kernel_resources import DMA_ALIGNMENT_BYTES, get_device_resources


@triton.jit
def _flatten_fused_overlap_metadata_kernel(
    token_to_req,
    q_cum,
    kv_lens,
    table,
    out_q,
    out_kv,
    out_table,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    META_TASKS: tl.constexpr,
    META_TILE: tl.constexpr,
    COPY_TILE: tl.constexpr,
):
    task = tl.program_id(0)
    if task < META_TASKS:
        rows = task * META_TILE + tl.arange(0, META_TILE)
        active_row = rows < ROWS
        row_req = tl.load(token_to_req + rows * REQ_STRIDE, active_row, other=0).to(tl.int64)
        end = tl.load(q_cum + row_req * Q_STRIDE, active_row, other=0).to(tl.int64)
        kv = tl.load(kv_lens + row_req * KV_STRIDE, active_row, other=0).to(tl.int64)
        # Cancel start algebraically, keeping the reference's int64 arithmetic.
        visible = kv - end + rows.to(tl.int64) + 1
        tl.store(out_q + rows, rows + 1, active_row)
        tl.store(out_kv + rows, visible.to(tl.int32), active_row)
    elif WIDTH > 0:
        begin = (task - META_TASKS).to(tl.int64) * COPY_TILE
        cols = tl.arange(0, COPY_TILE)
        dst = begin + cols
        if WIDTH % COPY_TILE == 0:
            copy_req = tl.load(token_to_req + (begin // WIDTH) * REQ_STRIDE).to(tl.int64)
            src = copy_req * TABLE_ROW_STRIDE + (begin % WIDTH + cols) * TABLE_COL_STRIDE
            tl.store(out_table + dst, tl.load(table + src))
        else:
            active = dst < ROWS * WIDTH
            flat_req = tl.load(token_to_req + (dst // WIDTH) * REQ_STRIDE, active, other=0).to(tl.int64)
            src = flat_req * TABLE_ROW_STRIDE + (dst % WIDTH) * TABLE_COL_STRIDE
            tl.store(out_table + dst, tl.load(table + src, active, other=0), active)


def flatten_fused_overlap_metadata(
    token_to_req: torch.Tensor,
    q_cum: torch.Tensor,
    kv_lens: torch.Tensor,
    table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand valid packed request metadata into one row per query token."""
    rows, width = token_to_req.numel(), table.shape[1]
    device = table.device
    out_q = torch.empty(rows, device=device, dtype=torch.int32)
    out_kv = torch.empty_like(out_q)
    out_table = torch.empty((rows, width), device=device, dtype=table.dtype)
    if rows == 0:
        return out_q, out_kv, out_table

    resources = get_device_resources(device.index)
    alignment = DMA_ALIGNMENT_BYTES // table.element_size()
    # Budget up to eight live 64-bit vectors per tile element for addresses,
    # values, and masks, then spread the flat PA copy across the vector cores.
    max_tile = 1 << ((resources.ub_bytes // (8 * 8)).bit_length() - 1)
    copy_lanes = max(1, triton.cdiv(rows * width, resources.vector_cores))
    copy_tile = min(max_tile, max(alignment, triton.next_power_of_2(copy_lanes)))
    meta_tile = min(max_tile, max(alignment, triton.next_power_of_2(rows)))
    meta_tasks = triton.cdiv(rows, meta_tile)
    _flatten_fused_overlap_metadata_kernel[(meta_tasks + triton.cdiv(rows * width, copy_tile),)](
        token_to_req,
        q_cum,
        kv_lens,
        table,
        out_q,
        out_kv,
        out_table,
        rows,
        width,
        token_to_req.stride(0),
        q_cum.stride(0),
        kv_lens.stride(0),
        *table.stride(),
        meta_tasks,
        meta_tile,
        copy_tile,
    )
    return out_q, out_kv, out_table
