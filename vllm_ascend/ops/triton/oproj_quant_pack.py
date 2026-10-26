# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize BF16 o_proj inputs directly into the DP AllToAll send layout.

The baseline torch_npu 2.10.0.post4 oracle was checked over all 65,280 finite
BF16 patterns with 6 synthetic and 79 model reciprocal/offset pairs: FP32
multiply then add matches exactly. NaN maps to 0; infinities saturate. Run
tests/ut/ops/test_oproj_quant_pack.py again when changing the target image.
Prepare stored BF16 parameters once after weight loading, before graph capture.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_pack_kernel(
    x_ptr,
    reciprocal_ptr,
    offset_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,
    OFFSET_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program writes a contiguous slice of one [destination, token] row.
    pid = tl.program_id(0)
    tiles_per_row = tl.cdiv(CHUNK, BLOCK)
    row = (pid // tiles_per_row) % N
    destination = pid // (tiles_per_row * N)
    column = (pid % tiles_per_row) * BLOCK + tl.arange(0, BLOCK)
    mask = column < CHUNK
    channel = destination * CHUNK + column
    x = tl.load(x_ptr + row * K + channel, mask=mask, other=0).to(tl.float32)
    if SCALE_SCALAR:
        reciprocal = tl.load(reciprocal_ptr).to(tl.float32)
    else:
        reciprocal = tl.load(reciprocal_ptr + channel, mask=mask, other=0).to(tl.float32)
    if OFFSET_SCALAR:
        offset = tl.load(offset_ptr).to(tl.float32)
    else:
        offset = tl.load(offset_ptr + channel, mask=mask, other=0).to(tl.float32)
    # Use the stored reciprocal directly. The launch disables contraction.
    product = x * reciprocal
    affine = product + offset
    # Observed npu_quantize mapping on the baseline image. Handle NaN before
    # min/max and float-to-int casts, whose NaN behavior is backend dependent.
    affine = tl.where(affine == affine, affine, 0.0)
    # NaNs are already zero, so propagating min/max avoids extra NaN repair.
    bounded = tl.minimum(
        tl.maximum(affine, -128.0, propagate_nan=tl.PropagateNan.ALL),
        127.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    # RNE without a backend-specific libdevice. Integer cast occurs only after
    # bounding and rounding, avoiding float-to-int truncation for finite data.
    # FP32 has unit spacing around 1.5 * 2**23; adding it rounds to even.
    RNE_MAGIC: tl.constexpr = 12582912.0
    rounded = (bounded + RNE_MAGIC) - RNE_MAGIC
    output_offset = (destination * N + row) * CHUNK + column
    tl.store(output_ptr + output_offset, rounded.to(tl.int8), mask=mask)


def prepare_parameters(scale_recip, offset):
    """Widen stored BF16 values once outside forward/capture; caller holds both copies.

    This is an explicit preparation API, not a lazy path in quantize_pack.
    It does not recompute the reciprocal or inspect/alter parameter values.
    Source parameters must stay alive until any old captured graphs are retired.
    """
    for name, parameter in (("scale_recip", scale_recip), ("offset", offset)):
        if (parameter.dtype != torch.bfloat16 or parameter.device.type != "npu"
                or not parameter.is_contiguous()):
            raise ValueError(f"{name} preparation requires contiguous stored BF16 on NPU")
    if scale_recip.device != offset.device:
        raise ValueError("Stored reciprocal and offset must be on the same device")
    return scale_recip.to(torch.float32), offset.to(torch.float32)


def quantize_pack(
    x: torch.Tensor,
    scale_recip: torch.Tensor,
    offset: torch.Tensor,
    exchange_size: int = 2,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """Return contiguous INT8 [exchange_size, N, K/exchange_size].

    Parameters are preprepared FP32 copies of stored BF16 values, either
    scalar or K-element vectors. For a known per-tensor quantizer whose
    parameters were expanded only to a weight shard, pass the [:1] views.
    No device-to-host value checks or reciprocal recomputation occur here.
    """
    if x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("quantize_pack requires a contiguous BF16 [N,K] input")
    if x.device.type != "npu":
        raise ValueError("quantize_pack requires an NPU input")
    n, k = x.shape
    if not isinstance(exchange_size, int) or exchange_size < 1 or k < 1 or k % exchange_size:
        raise ValueError("exchange_size must be positive and divide the input width")
    if not isinstance(block_size, int) or block_size < 32 or block_size > 4096 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two between 32 and 4096")
    for name, parameter in (("scale_recip", scale_recip), ("offset", offset)):
        if parameter.device != x.device or parameter.dtype != torch.float32 or not parameter.is_contiguous():
            raise ValueError(f"{name} must be contiguous FP32 on the input device")
        if parameter.numel() not in (1, k):
            raise ValueError(f"{name} must contain one or K={k} values")
    chunk = k // exchange_size
    output = torch.empty((exchange_size, n, chunk), device=x.device, dtype=torch.int8)
    if n:
        grid = (exchange_size * n * triton.cdiv(chunk, block_size),)
        _quantize_pack_kernel[grid](
            x, scale_recip, offset, output,
            N=n, K=k, CHUNK=chunk,
            SCALE_SCALAR=scale_recip.numel() == 1,
            OFFSET_SCALAR=offset.numel() == 1,
            BLOCK=block_size,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output
