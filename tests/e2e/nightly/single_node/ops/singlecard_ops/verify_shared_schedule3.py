# SPDX-License-Identifier: Apache-2.0
"""Standalone NPU shared-math/stream smoke test; not an EP fused-MC2 test."""

from types import SimpleNamespace as NS

import torch
import torch_npu
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.shared_schedule import run_scheme3
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op, maybe_trans_nz


@torch.inference_mode()
def main():
    if not torch.npu.is_available():
        raise RuntimeError("NPU validation requires a real Ascend device")
    torch.npu.set_device(0)
    init_ascend_config(VllmConfig(additional_config={"weight_nz_mode": 1}))
    print("smoke_config: weight_nz_mode=1 logical_device=0 routed_mc2=stub", flush=True)
    if not enable_custom_op():
        raise RuntimeError("Required vLLM Ascend custom operators are unavailable")
    torch.manual_seed(42)
    torch_npu.npu.config.allow_internal_format = True
    x = torch.randn(6, 256, device="npu", dtype=torch.bfloat16)

    def projection(n, m):
        scale = torch.full((m,), 0.01, device="npu", dtype=torch.float32)
        return NS(
            weight=maybe_trans_nz(torch.randint(-16, 16, (n, m), device="npu", dtype=torch.int8)),
            weight_scale=scale,
            weight_scale_fp32=scale,
        )

    owner = NS(
        quant_type=QuantType.W8A8,
        routed_experts=NS(swiglu_limit=0.0),
        _shared_experts=NS(gate_up_proj=projection(256, 512), down_proj=projection(256, 256)),
    )
    gate_weight = torch.randn(256, 256, device="npu", dtype=torch.float32)
    stream = torch.npu.Stream()

    def gate():
        return x.float() @ gate_weight

    def route(logits, callback):
        # Real TopK; the routed FFN below is synthetic and has no EP communication.
        weights = logits.topk(8, dim=-1).values
        callback()
        return weights.sum(dim=-1)

    def steps():
        return AscendMoERunner._scheduled_shared_steps(owner, x)

    def scheduled():
        return run_scheme3(torch.npu, stream, x, steps(), gate, route)

    for _ in range(3):
        scheduled()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual, actual_route = scheduled()
    # Replay with changed values catches stale cross-stream graph dependencies.
    for _ in range(3):
        x.normal_()
        reference_graph = torch.npu.NPUGraph()
        with torch.npu.graph(reference_graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            expected = list(steps())[-1]
            expected_route = route(gate(), lambda: None)
        reference_graph.replay()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_route, expected_route, rtol=0, atol=0)
    print("PASS: shared INT8 math, stream joins, changed-input NPUGraph replay; fused MC2/EP not tested")


if __name__ == "__main__":
    main()
