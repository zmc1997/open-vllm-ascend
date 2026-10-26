# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in o_proj collectives on the caller's computation stream.

Construct once after the DP2 exchange and TP4 total groups exist, and reuse
across layers. The owner must release captured graphs before calling close().
"""

import ctypes

import torch
import torch.distributed as dist

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataType_t,
    hcclDataTypeEnum,
    hcclResult_t,
)


class OProjNativeCommunicator:
    def __init__(self, exchange_group, total_group, device):
        for group, size in ((exchange_group, 2), (total_group, 4)):
            if group.world_size != size or dist.get_world_size(group.cpu_group) != size:
                raise ValueError(f"o_proj native communication requires group size {size}")
            if dist.get_backend(group.cpu_group) != dist.Backend.GLOO:
                raise ValueError("o_proj native communication requires existing Gloo CPU groups")
        self.device = torch.device(f"npu:{device}" if isinstance(device, int) else device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("o_proj native communication requires an indexed NPU device")
        self._exchange_cpu_group = exchange_group.cpu_group
        self._total_cpu_group = total_group.cpu_group
        self._closed = False
        self._close_error = None
        # Match the verified micro's collective initialization order. Never
        # touch either coordinator's device_group or initialize per layer.
        self._total = PyHcclCommunicator(self._total_cpu_group, self.device)
        self._require_enabled(self._total)
        # Public CANN 9.1 hccl.h ABI, verified on the target image: send/recv
        # counts are uint64 elements PER PEER. Bind a private callable to the
        # installed symbol; no runtime header path or shared wrapper mutation.
        signature = ctypes.CFUNCTYPE(
            hcclResult_t, buffer_type, ctypes.c_uint64, hcclDataType_t,
            buffer_type, ctypes.c_uint64, hcclDataType_t, hcclComm_t, aclrtStream_t,
        )
        self._alltoall = signature(("HcclAlltoAll", self._total.hccl.lib))
        self._exchange = PyHcclCommunicator(self._exchange_cpu_group, self.device)
        self._require_enabled(self._exchange)

    def _require_enabled(self, communicator):
        if self._closed or not communicator.available or communicator.disabled:
            raise RuntimeError("o_proj native communicator is unavailable or disabled; fallback is forbidden")

    def all_to_all(self, send_flat: torch.Tensor, recv_flat: torch.Tensor) -> None:
        self._require_enabled(self._exchange)
        for tensor in (send_flat, recv_flat):
            if (tensor.device != self.device or tensor.dtype != torch.int8
                    or tensor.ndim != 1 or not tensor.is_contiguous()):
                raise ValueError("o_proj A2A requires contiguous flat INT8 tensors on its NPU device")
        count = send_flat.numel()
        if count == 0 or count % 2 or recv_flat.numel() != count:
            raise ValueError("o_proj A2A requires equal nonempty buffers divisible between two peers")
        send_ptr, recv_ptr = send_flat.data_ptr(), recv_flat.data_ptr()
        if send_ptr < recv_ptr + count and recv_ptr < send_ptr + count:
            raise ValueError("o_proj A2A send and receive buffers must not overlap")
        dtype = hcclDataTypeEnum.from_torch(torch.int8)
        stream = torch.npu.current_stream(self.device)
        self._exchange.hccl.HCCL_CHECK(self._alltoall(
            buffer_type(send_ptr), count // 2, dtype,
            buffer_type(recv_ptr), count // 2, dtype,
            self._exchange.comm, aclrtStream_t(stream.npu_stream),
        ))

    def all_reduce(self, partial: torch.Tensor) -> torch.Tensor:
        self._require_enabled(self._total)
        if (partial.device != self.device or partial.dtype != torch.bfloat16
                or partial.ndim != 2 or not partial.is_contiguous() or partial.numel() == 0):
            raise ValueError("o_proj AR4 requires a nonempty contiguous BF16 matrix on its NPU device")
        output = self._total.all_reduce(partial, stream=torch.npu.current_stream(self.device))
        if output is None:
            raise RuntimeError("o_proj native AR4 returned no output; fallback is forbidden")
        return output

    def close(self) -> None:
        """Collective shutdown after graph release; do not call from __del__."""
        if self._close_error is not None:
            raise RuntimeError(
                "o_proj native communicator shutdown previously failed; recreate the process"
            ) from self._close_error
        if self._closed:
            return
        self._exchange.disabled = True
        self._total.disabled = True
        try:
            torch.npu.synchronize(self.device)
            dist.barrier(group=self._exchange_cpu_group)
            dist.barrier(group=self._total_cpu_group)
            try:
                self._exchange.hccl.hcclCommDestroy(self._exchange.comm)
            finally:
                self._total.hccl.hcclCommDestroy(self._total.comm)
        except Exception as exc:
            self._close_error = exc
            raise
        self._closed = True
