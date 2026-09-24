# SPDX-License-Identifier: Apache-2.0
"""NPU smoke for actual shared math and scheduling (routed MC2 is stubbed).

Run against the installed patched package. Missing dependencies/NPU fail the test.
This does not validate distributed DispatchFFNCombine, model accuracy or speedup.
"""

import argparse
from types import SimpleNamespace as NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.iterations) <= 0:
        parser.error("dimensions and iterations must be positive")

    import torch
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, FusedMoEEvents
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import enable_custom_op, maybe_trans_nz

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; this smoke test was NOT validated")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    init_ascend_config(VllmConfig(additional_config={"weight_nz_mode": 1}))
    print("smoke_config: weight_nz_mode=1 logical_device=0 routed_mc2=stub", flush=True)
    if not enable_custom_op():
        raise RuntimeError("Ascend custom operators unavailable; this smoke test was NOT validated")
    torch.manual_seed(19)
    device, dtype = "npu:0", torch.bfloat16

    def projection(k, n):
        scale = torch.full((n,), 0.01, dtype=dtype, device=device)
        return NS(
            weight=maybe_trans_nz(torch.randint(-8, 8, (k, n), dtype=torch.int8, device=device)),
            weight_scale=scale,
            weight_scale_fp32=scale.float(),
        )

    shared = NS(
        gate_up_proj=projection(args.hidden, 2 * args.intermediate),
        down_proj=projection(args.intermediate, args.hidden),
        act_fn=object(),
    )
    runner = NS(
        _shared_experts=shared,
        gate=NS(weight_fp32=torch.randn(256, args.hidden, device=device, dtype=torch.float32) * 0.01),
        routed_experts=NS(swiglu_limit=0.0, swiglu_alpha=1.0, swiglu_beta=0.0),
        multistream_overlap_shared_expert=False,
        _prepare_shared_expert_input=lambda x: x,
        _finalize_shared_expert_output=lambda x: x,
    )

    def routed_stub(hidden, logits, before_fused_experts=None):
        # Execute a real TopK, then the exact integration callback. This is
        # intentionally a single-device stub, not DispatchFFNCombine.
        weights, _ = torch.topk(logits, 8, dim=-1)
        if before_fused_experts is not None:
            before_fused_experts()
        return hidden + weights.sum(dim=-1, keepdim=True).to(hidden.dtype)

    runner.no_shared_forward_impl = routed_stub
    with torch.inference_mode():
        for quant_type in (QuantType.W8A8, QuantType.W4A8):
            runner.quant_type = quant_type
            for _ in range(args.iterations):
                hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
                ready = torch.npu.current_stream().record_event()
                baseline = AscendMoERunner._forward_shared_experts(
                    runner, hidden, FusedMoEEvents(before_routed_experts=ready)
                )
                actual, actual_routed = AscendMoERunner._fused_mc2_scheduled_forward(runner, hidden, hidden)
                expected_routed = routed_stub(
                    hidden, torch.nn.functional.linear(hidden.float(), runner.gate.weight_fp32)
                )
                torch.npu.synchronize()
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                torch.testing.assert_close(actual_routed, expected_routed, rtol=0, atol=0)
    print("PASS: actual shared NPU operators and scheduled output match baseline; routed MC2 was stubbed.")


if __name__ == "__main__":
    main()
