# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strided ND V-up BMM; new shape/dtype domains require the strict NPU oracle."""

import math
import os
from ctypes import c_float
from functools import cache
from hashlib import sha256
from pathlib import Path

import torch
import torch_npu
import triton.language.extra.cann.extension as al
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.kernel_resources import DMA_ALIGNMENT_BYTES, DeviceResources, get_device_resources


@al.register_custom_op
class _vup_hf32_enable:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_S
    symbol = "vup_hf32_enable"
    bitcode = str(Path(__file__).with_name("vup_hf32_control.bc"))


@al.register_custom_op
class _vup_hf32_disable:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_S
    symbol = "vup_hf32_disable"
    bitcode = str(Path(__file__).with_name("vup_hf32_control.bc"))


@cache
def _hf32_control_sha() -> str:
    """Include external Cube code in the JIT key; read it once per process."""
    return sha256(Path(_vup_hf32_enable.bitcode).read_bytes()).hexdigest()


def _cube_math_type() -> int:
    """Observe the original torch-npu override/precision rules without setters."""
    override = torch_npu.npu.matmul.cube_math_type
    if override is not None:
        return override.value
    precision = torch_npu._C._npu_getOption("ACL_PRECISION_MODE")
    if isinstance(precision, bytes):
        precision = precision.decode()
    down_precision = torch_npu._C._npu_get_soc_version() < 220
    if precision == "allow_fp32_to_fp16":
        down_precision = True
    elif precision == "must_keep_origin_dtype":
        down_precision = False
    code = (int(torch_npu.npu.matmul.allow_hf32) << 1) | int(down_precision)
    return (0, 2, 3, 1)[code]


@cache
def _fp32_hardware(device_index: int) -> tuple[int, int, int]:
    """Read the same platform capacities used by CANN's host tiling, once."""
    resources = get_device_resources(device_index)
    return resources.cube_cores, resources.l0c_bytes, resources.l2_bytes


def _fp32_tiling(n: int, h: int, k: int, v: int, cores: int, l0c: int, l2: int) -> dict[str, int]:
    """CANN 9.1 PpMatMulDefault ND/no-transpose-B tiling, using actual dimensions.

    Source: cann/ops-nn v9.1.0, transpose_batch_mat_mul/op_host/op_tiling/
    pp_matmul_default.cpp and pp_matmul_common_tiling.h. K shuffle also depends
    on the original core mapping; it is not a fixed rotation of half the columns.
    """
    ceildiv = lambda a, b: (a + b - 1) // b
    roundup = lambda a, b: ceildiv(a, b) * b
    f32 = lambda value: c_float(value).value
    primary_m = n >= v
    # Preserve the host expression's natural logarithm, including tie ordering.
    round_base = (1 << math.ceil(math.log(ceildiv(v if primary_m else n, 16)))) * 16
    primary = roundup(n if primary_m else v, 16)
    secondary = roundup(v if primary_m else n, round_base)
    best_cost = 1.0
    m0 = n0 = 16
    for first in (16, 32, 64, 128):
        if first > primary:
            break
        for second in (16, 32, 64, 128):
            if second > secondary or first * second * 4 > l0c:
                break
            candidate_m, candidate_n = (first, second) if primary_m else (second, first)
            m_loop, n_loop = ceildiv(n, candidate_m), ceildiv(v, candidate_n)
            block_dim = min(h * m_loop * n_loop, cores)
            m_once = candidate_m if block_dim < n_loop else block_dim // n_loop * candidate_m
            n_once = cores * candidate_n if block_dim < n_loop else v
            a_coef = 5 if f32(m_once * k * 4) > f32(l2) else 1
            b_coef = 5 if f32(n_once * k * 4) > f32(l2) else 1
            cost = f32(f32(1 / (a_coef * candidate_n)) + f32(1 / (b_coef * candidate_m)))
            if cost < best_cost:
                best_cost, m0, n0 = cost, candidate_m, candidate_n
    m_loop, n_loop = ceildiv(n, m0), ceildiv(v, n0)
    total = h * m_loop * n_loop
    block_dim = min(total, cores)
    minimum = f32(f32(f32(n) * f32(k)) + f32(f32(k) * f32(v)))
    swizzle, direction = 1, 0
    for count in range(1, block_dim + 1):
        columns = ceildiv(block_dim, count)
        if f32(count * n0 + n) < f32(m0 * columns + v):
            direction = 1
            cost = f32(n0 * count + m0 * columns)
            if cost <= minimum:
                minimum, swizzle = cost, count
        else:
            direction = 0
            cost = f32(m0 * count + n0 * columns)
            if cost < minimum:
                minimum, swizzle = cost, count
    # Keep this original CANN 9.1 PP algorithm budget: changing native K tiles
    # changes its core-dependent K shuffle and therefore FP32 rounding order.
    # It is L1AB_PINGPONG_BUFFER_SIZE in pp_matmul_common_tiling.h, not a tensor size.
    k_max = int(f32(262144 / ((m0 + n0) * 4)))
    k0 = k_max // (16 if k_max < 256 else 256) * (16 if k_max < 256 else 256)
    if k0 > 512:
        k0 = k0 // 512 * 512
    return dict(
        BLOCK_M=m0,
        BLOCK_V=n0,
        K_TILE=k0,
        M_LOOP=m_loop,
        N_LOOP=n_loop,
        CORE_COUNT=block_dim,
        SWIZZLE=swizzle,
        SWIZZLE_DIRECTION=direction,
        TOTAL=total,
    )


@cache
def _half_tiling(
    n: int, h: int, k: int, v: int, element_bytes: int, resources: DeviceResources
) -> tuple[int, int, int]:
    """CANN PP M/N search and ping-pong capacity bounds for FP16/BF16.

    The 16-element Cube alignment, 512-byte candidate-axis limit and the 5:1
    L2/HBM cost coefficients come from CANN 9.1 pp_matmul_common_tiling.h and
    pp_matmul_info.h. Actual capacities and core counts come from this device.
    K remains in ascending order, with FP32 accumulation and one final cast.
    Changed dot tiles require the same strict NPU oracle gate as a new kernel.
    """
    alignment = DMA_ALIGNMENT_BYTES // element_bytes
    ceildiv = lambda a, b: (a + b - 1) // b
    roundup = lambda a, b: ceildiv(a, b) * b
    f32 = lambda value: c_float(value).value
    primary_m = n >= v
    round_base = (1 << math.ceil(math.log(ceildiv(v if primary_m else n, alignment)))) * alignment
    primary = roundup(n if primary_m else v, alignment)
    secondary = roundup(v if primary_m else n, round_base)
    axis_limit = 512 // element_bytes
    best_cost, block_m, block_v = 1.0, alignment, alignment
    first = alignment
    while first <= min(primary, axis_limit):
        second = alignment
        while second <= min(secondary, axis_limit):
            candidate_m, candidate_v = (first, second) if primary_m else (second, first)
            if candidate_m * candidate_v * 4 <= resources.l0c_bytes:
                m_loop, v_loop = ceildiv(n, candidate_m), ceildiv(v, candidate_v)
                active = min(h * m_loop * v_loop, resources.cube_cores)
                m_once = candidate_m if active < v_loop else active // v_loop * candidate_m
                v_once = resources.cube_cores * candidate_v if active < v_loop else v
                a_coef = 5 if f32(m_once * k * element_bytes) > f32(resources.l2_bytes) else 1
                b_coef = 5 if f32(v_once * k * element_bytes) > f32(resources.l2_bytes) else 1
                cost = f32(f32(1 / (a_coef * candidate_v)) + f32(1 / (b_coef * candidate_m)))
                if cost < best_cost:
                    best_cost, block_m, block_v = cost, candidate_m, candidate_v
            second *= 2
        first *= 2
    # This is the L1 operand tile: reserve one half of L1 per ping-pong stage,
    # as in CANN PP. Ascend's mma_tile_{half,bfloat16}_to_float lowering splits
    # that tile into its own L1-to-L0 k_part loop; its L0 microtile must not cap
    # the top-level tl.dot K, which would repeat GM staging for every microtile.
    k_limit = resources.l1_bytes // (2 * (block_m + block_v) * element_bytes)
    # Round the actual K extent up; the capacity bound remains rounded down.
    block_k = min(1 << (max(alignment, k) - 1).bit_length(), 1 << (k_limit.bit_length() - 1))
    return block_m, block_k, block_v


def _copy_block(
    size: int, element_bytes: int, resources: DeviceResources, stride: int = 1, width: int | None = None
) -> int:
    """Balance actual row work and account for strided DMA or span gathering."""
    per_core = triton.cdiv(size, resources.vector_cores)
    block = min(triton.next_power_of_2(width or size), triton.next_power_of_2(per_core))
    while True:
        span = triton.next_power_of_2((block - 1) * stride + 1)
        # A short-stride span is loaded once and selected inside UB. Otherwise
        # the installed backend stages a DMA block for each strided lane.
        payload = min(span * element_bytes, block * DMA_ALIGNMENT_BYTES) if stride > 1 else block * element_bytes
        working = payload + block * (element_bytes + 2 * 8 + 1)
        if working <= resources.ub_bytes:
            return block
        if block == 1:
            raise RuntimeError("Device UB cannot hold one bit-copy tile")
        block //= 2


def _copy_tiling(shape, strides, element_bytes, resources):
    """Plan dense suffix gathering or a transpose along the physical unit axis."""
    size = math.prod(shape)
    axis = next((i for i in range(2) if shape[i] > 1 and strides[i] == 1), None)
    if axis is not None and strides[2] > 0:
        inner = min(triton.next_power_of_2(shape[axis]), resources.cache_line_bytes // element_bytes)
        per_core = triton.cdiv(size, resources.vector_cores)
        outer = min(triton.next_power_of_2(shape[2]), triton.next_power_of_2(triton.cdiv(per_core, inner)))
        alignment = DMA_ALIGNMENT_BYTES // element_bytes
        while (
            2 * triton.cdiv(inner, alignment) * alignment * triton.cdiv(outer, alignment) * alignment * element_bytes
            + (inner + outer) * (2 * 8 + 1)
            > resources.ub_bytes
        ):
            if outer > 1:
                outer //= 2
            elif inner > 1:
                inner //= 2
            else:
                raise RuntimeError("Device UB cannot hold a DMA-aligned transpose tile")
        return axis, inner, outer
    # Collapse only a genuinely linear logical suffix; row gaps/broadcast
    # axes otherwise retain their original scalar address calculation.
    first, width = 2, shape[2]
    for dimension in (1, 0):
        if shape[dimension] != 1 and strides[dimension] != width * strides[2]:
            break
        first, width = dimension, width * shape[dimension]
    return first + 3, width, _copy_block(size, element_bytes, resources, strides[2], width)


@triton.jit
def _vup_fp32_bmm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    K_TILE: tl.constexpr,
    M_LOOP: tl.constexpr,
    N_LOOP: tl.constexpr,
    CORE_COUNT: tl.constexpr,
    SWIZZLE: tl.constexpr,
    SWIZZLE_DIRECTION: tl.constexpr,
    SHUFFLE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    MATH_MODE: tl.constexpr = 0,
    CONTROL_SHA: tl.constexpr = "",
):
    # This runtime's generic HF32 meta-op controls the wrong register bit on
    # A3. Use the original CANN Cube-local controls; no process setting changes.
    if MATH_MODE == 3:
        al.custom("_vup_hf32_enable")
    tile = tl.program_id(0).to(tl.int64)
    head = tile // (M_LOOP * N_LOOP)
    in_batch = tile % (M_LOOP * N_LOOP)
    if SWIZZLE_DIRECTION == 0:
        group = in_batch // (SWIZZLE * N_LOOP)
        within = in_batch % (SWIZZLE * N_LOOP)
        rows = tl.minimum(SWIZZLE, M_LOOP - group * SWIZZLE)
        m_index = group * SWIZZLE + within % rows
        n_index = within // rows
        n_index = tl.where(group % 2 == 0, n_index, N_LOOP - n_index - 1)
    else:
        group = in_batch // (SWIZZLE * M_LOOP)
        within = in_batch % (SWIZZLE * M_LOOP)
        columns = tl.minimum(SWIZZLE, N_LOOP - group * SWIZZLE)
        m_index = within // columns
        n_index = group * SWIZZLE + within % columns
        m_index = tl.where(group % 2 == 0, m_index, M_LOOP - m_index - 1)
    row = m_index * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    column = n_index * BLOCK_V + tl.arange(0, BLOCK_V).to(tl.int64)
    shift = (tile % CORE_COUNT) % tl.cdiv(K, K_TILE) if SHUFFLE_K else 0
    accumulator = tl.full((BLOCK_M, BLOCK_V), 0, tl.float32)
    for iteration in range(tl.cdiv(K, K_TILE)):
        k_block = (iteration + shift) % tl.cdiv(K, K_TILE)
        for start in range(0, K_TILE, BLOCK_K):
            # Do not merge a short final native K tile into the following tile.
            # That would change the Cube reduction groups after K wraps around.
            if k_block * K_TILE + start < K:
                reduction = k_block * K_TILE + start + tl.arange(0, BLOCK_K).to(tl.int64)
                valid_k = (reduction < K) & (reduction < (k_block + 1) * K_TILE)
                x = tl.load(
                    x_ptr + (row[:, None] * H + head) * K + reduction[None, :],
                    mask=(row[:, None] < N) & valid_k[None, :],
                    other=0,
                )
                weight = tl.load(
                    weight_ptr + (head * K + reduction[:, None]) * V + column[None, :],
                    mask=valid_k[:, None] & (column[None, :] < V),
                    other=0,
                )
                # The enclosing Cube control selects HF32 when requested.
                # The IEEE meta-op leaves it intact and avoids the faulty
                # generic HF32 control helper in this compiler version.
                accumulator = tl.dot(x, weight, accumulator, input_precision="ieee", out_dtype=tl.float32)
    tl.store(
        output_ptr + (row[:, None] * H + head) * V + column[None, :],
        accumulator,
        mask=(row[:, None] < N) & (column[None, :] < V),
    )
    if MATH_MODE == 3:
        al.custom("_vup_hf32_disable")


@triton.jit
def _vup_bmm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    X_N: tl.constexpr,
    X_H: tl.constexpr,
    X_K: tl.constexpr,
    W_H: tl.constexpr,
    W_K: tl.constexpr,
    W_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    head_tile = tl.program_id(0)
    head = (head_tile // tl.cdiv(V, BLOCK_V)).to(tl.int64)
    column = ((head_tile % tl.cdiv(V, BLOCK_V)) * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    row = (tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    reduction = tl.arange(0, BLOCK_K).to(tl.int64)
    x = tl.load(
        x_ptr + row[:, None] * X_N + head * X_H + reduction[None, :] * X_K,
        mask=(row[:, None] < N) & (reduction[None, :] < K),
        other=0,
    )
    weight = tl.load(
        weight_ptr + head * W_H + reduction[:, None] * W_K + column[None, :] * W_V,
        mask=(reduction[:, None] < K) & (column[None, :] < V),
        other=0,
    )
    # Keep one ascending FP32 accumulation chain across L1 operand tiles.
    accumulator = tl.dot(x, weight, out_dtype=tl.float32)
    for start in range(BLOCK_K, K, BLOCK_K):
        reduction = start + tl.arange(0, BLOCK_K).to(tl.int64)
        x = tl.load(
            x_ptr + row[:, None] * X_N + head * X_H + reduction[None, :] * X_K,
            mask=(row[:, None] < N) & (reduction[None, :] < K),
            other=0,
        )
        weight = tl.load(
            weight_ptr + head * W_H + reduction[:, None] * W_K + column[None, :] * W_V,
            mask=(reduction[:, None] < K) & (column[None, :] < V),
            other=0,
        )
        accumulator = tl.dot(x, weight, accumulator, out_dtype=tl.float32)
    result = accumulator.to(output_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(
        output_ptr + (row[:, None] * H + head) * V + column[None, :],
        result,
        mask=(row[:, None] < N) & (column[None, :] < V),
    )


@triton.jit
def _vup_contiguous_copy_kernel(
    source,
    destination,
    D1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    MERGED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    DMA_BYTES: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    begin = tl.program_id(0).to(tl.int64) * BLOCK
    lanes = tl.arange(0, BLOCK)
    if MERGED == 0:
        base = tl.full((), 0, tl.int64)
    elif MERGED == 1:
        base = row * S0
    else:
        base = (row // D1) * S0 + (row % D1) * S1
    origin = source + base + begin * S2
    if S2 > 1 and SPAN * ELEMENT_BYTES < BLOCK * DMA_BYTES:
        physical = tl.arange(0, SPAN)
        count = tl.minimum(BLOCK, WIDTH - begin)
        values = tl.load(origin + physical, physical < (count - 1) * S2 + 1, other=0)
        # A3 gather accepts floating carriers but not raw INT16/INT32. Bitcast
        # only: arithmetic conversion would lose NaN payloads and signed zero.
        if ELEMENT_BYTES == 2:
            carrier = values.to(tl.float16, bitcast=True)
        else:
            carrier = values.to(tl.float32, bitcast=True)
        value = tl.gather(carrier, lanes * S2, axis=0).to(source.dtype.element_ty, bitcast=True)
    else:
        if (BLOCK - 1) * S2 > (1 << 31) - 1:
            offsets = lanes.to(tl.int64) * S2
        else:
            offsets = lanes * S2
        value = tl.load(origin + offsets, begin + lanes < WIDTH, other=0)
    tl.store(destination + row * WIDTH + begin + lanes, value, begin + lanes < WIDTH)


@triton.jit
def _vup_transpose_copy_kernel(
    source,
    destination,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    OUTER_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_INNER_STRIDE: tl.constexpr,
    INNER_BLOCK: tl.constexpr,
    OUTER_BLOCK: tl.constexpr,
    FULL: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * INNER_BLOCK
    last = tl.program_id(1).to(tl.int64) * OUTER_BLOCK
    batch = tl.program_id(2).to(tl.int64)
    inner = tl.arange(0, INNER_BLOCK)
    outer = tl.arange(0, OUTER_BLOCK)
    source_origin = source + batch * BATCH_STRIDE + last * OUTER_STRIDE + first
    target_origin = destination + batch * OUTPUT_BATCH_STRIDE + first * OUTPUT_INNER_STRIDE + last
    if (OUTER_BLOCK - 1) * OUTER_STRIDE + INNER_BLOCK > (1 << 31) - 1:
        source_outer = outer.to(tl.int64)
    else:
        source_outer = outer
    if (INNER_BLOCK - 1) * OUTPUT_INNER_STRIDE + OUTER_BLOCK > (1 << 31) - 1:
        target_inner = inner.to(tl.int64)
    else:
        target_inner = inner
    source_tile = source_origin + source_outer[:, None] * OUTER_STRIDE + inner[None, :]
    target_tile = target_origin + target_inner[:, None] * OUTPUT_INNER_STRIDE + outer[None, :]
    if FULL:
        value = tl.load(source_tile)
        tl.store(target_tile, tl.trans(value))
    else:
        inner_valid = inner < INNER - first
        outer_valid = outer < OUTER - last
        value = tl.load(source_tile, outer_valid[:, None] & inner_valid[None, :], other=0)
        # Form the target mask directly. A transposed boolean mask caused
        # destination read/modify/write and scalar conversion in the A3 backend.
        tl.store(target_tile, tl.trans(value), inner_valid[:, None] & outer_valid[None, :])


def _contiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Bit-copy ND views; keep all matrix arithmetic after dense normalization."""
    result = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    bit_dtype = torch.int16 if tensor.element_size() == 2 else torch.int32
    resources = get_device_resources(tensor.device.index)
    mode, first, second = _copy_tiling(tensor.shape, tensor.stride(), tensor.element_size(), resources)
    if mode < 3:
        axis, batch_axis = mode, 1 - mode
        inner, outer = tensor.shape[axis], tensor.shape[2]
        _vup_transpose_copy_kernel[(triton.cdiv(inner, first), triton.cdiv(outer, second), tensor.shape[batch_axis])](
            tensor.view(bit_dtype),
            result.view(bit_dtype),
            INNER=inner,
            OUTER=outer,
            BATCH_STRIDE=tensor.stride(batch_axis),
            OUTER_STRIDE=tensor.stride(2),
            OUTPUT_BATCH_STRIDE=result.stride(batch_axis),
            OUTPUT_INNER_STRIDE=result.stride(axis),
            INNER_BLOCK=first,
            OUTER_BLOCK=second,
            FULL=inner % first == 0 and outer % second == 0,
            multibuffer=False,
        )
    else:
        _vup_contiguous_copy_kernel[(triton.cdiv(first, second), tensor.numel() // first)](
            tensor.view(bit_dtype),
            result.view(bit_dtype),
            D1=tensor.shape[1],
            S0=tensor.stride(0),
            S1=tensor.stride(1),
            S2=tensor.stride(2),
            MERGED=mode - 3,
            WIDTH=first,
            BLOCK=second,
            SPAN=triton.next_power_of_2((second - 1) * tensor.stride(2) + 1),
            ELEMENT_BYTES=tensor.element_size(),
            DMA_BYTES=DMA_ALIGNMENT_BYTES,
            multibuffer=False,
        )
    return result


def vup_bmm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[N,H,K] x [H,K,V] -> [N,H,V], in the input dtype, with no CANN fallback.

    Ordinary ND views may have positive or zero strides. Non-ND and empty inputs
    are outside the original aclnnTransposeBatchMatMul domain on this runtime.
    BF16, FP16 and FP32 lowering/bit equality are gated by the NPU driver.
    """
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or x.ndim != 3
        or weight.ndim != 3
        or x.shape[1:] != weight.shape[:2]
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or weight.dtype != x.dtype
        or x.device.type != "npu"
        or weight.device != x.device
        or any(stride < 0 for stride in (*x.stride(), *weight.stride()))
    ):
        raise ValueError("V-up BMM requires matching ND [N,H,K] and [H,K,V] tensors on one NPU")
    n, h, k = x.shape
    v = weight.shape[2]
    if not n or not h or not k or not v:
        raise RuntimeError("V-up BMM does not support empty tensors, matching aclnnTransposeBatchMatMul")
    if torch_npu.get_npu_format(x) != 2 or torch_npu.get_npu_format(weight) != 2:
        raise ValueError("V-up BMM requires ND input tensors")
    # Avoid implicit strided operand gathering in the dot lowering. Keep the
    # contiguous path unchanged; the separate copy preserves every source bit.
    if not x.is_contiguous():
        x = _contiguous_copy(x)
    if not weight.is_contiguous():
        weight = _contiguous_copy(weight)
    output = torch.empty((n, h, v), device=x.device, dtype=x.dtype)
    if x.dtype == torch.float32:
        math_mode = _cube_math_type()
        plan = _fp32_tiling(n, h, k, v, *_fp32_hardware(x.device.index))
        total = plan.pop("TOTAL")
        # TBMM 9.1's actual modes 0/1/2 are IEEE (no FP16 input cast); mode 3
        # enables HF32 and leaves the PP branch, which owns this K shuffle.
        # Its API routes mode 4 to mode 0 on the target DAV_2201 architecture.
        shuffle = (
            math_mode != 3
            and os.getenv("CLOSE_MATMUL_K_SHIFT") != "1"
            and not (torch.are_deterministic_algorithms_enabled() and torch_npu._C._npu_get_deterministic_level() > 1)
        )
        _vup_fp32_bmm_kernel[(total,)](
            x,
            weight,
            output,
            N=n,
            H=h,
            K=k,
            V=v,
            **plan,
            SHUFFLE_K=shuffle,
            MATH_MODE=math_mode,
            BLOCK_K=min(512, plan["K_TILE"]),
            enable_fp_fusion=False,
            CONTROL_SHA=_hf32_control_sha() if math_mode == 3 else "",
        )
        return output
    block_m, block_k, block_v = _half_tiling(n, h, k, v, x.element_size(), get_device_resources(x.device.index))
    _vup_bmm_kernel[(h * triton.cdiv(v, block_v), triton.cdiv(n, block_m))](
        x,
        weight,
        output,
        N=n,
        H=h,
        K=k,
        V=v,
        X_N=x.stride(0),
        X_H=x.stride(1),
        X_K=x.stride(2),
        W_H=weight.stride(0),
        W_K=weight.stride(1),
        W_V=weight.stride(2),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        enable_fp_fusion=False,
    )
    return output
