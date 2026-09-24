# SPDX-License-Identifier: Apache-2.0
"""Real Gate constructor/weight-loader regression; requires one available NPU.

This checks router readiness and logits, not distributed MC2 or model accuracy.
No Gate constructor, quant method or weight tensor is mocked/replaced.
"""

from types import SimpleNamespace


def main():
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.utils.network_utils import get_open_port

    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; real Gate loader was NOT validated")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    device = torch.device("npu:0")
    torch.manual_seed(19)
    initial_config = VllmConfig()
    init_ascend_config(initial_config)
    try:
        with set_current_vllm_config(initial_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                backend="hccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        for enabled, bias in ((False, False), (True, False), (True, True)):
            config = VllmConfig(additional_config={"weight_nz_mode": 1, "fused_mc2_shared_schedule": enabled})
            init_ascend_config(config)
            with set_current_vllm_config(config), torch.inference_mode():
                gate = AscendGateLinear(6144, 256, bias=bias, prefix="model.layers.0.mlp.gate").to(device)
                scheduled = enabled and not bias
                expected_method = AscendUnquantizedLinearMethod if scheduled else UnquantizedLinearMethod
                assert type(gate.quant_method) is expected_method, type(gate.quant_method)
                assert not hasattr(gate, "weight_fp32")
                weight = torch.randn(256, 6144, device=device, dtype=torch.float32) * 0.01
                gate.weight_loader(gate.weight, weight)
                if bias:
                    gate.weight_loader(gate.bias, torch.zeros(256, device=device))
                # Invoke the method the real constructor selected, just as the loader does.
                gate.quant_method.process_weights_after_loading(gate)
                assert hasattr(gate, "weight_fp32") is scheduled
                runner = SimpleNamespace(gate=gate, fused_mc2_shared_schedule=enabled, _shared_experts=object())
                assert AscendMoERunner.is_internal_router.fget(runner) is scheduled
                if scheduled:
                    assert gate.weight_fp32.dtype == torch.float32
                    hidden = torch.randn(6, 6144, device=device, dtype=torch.bfloat16)
                    expected = torch.nn.functional.linear(hidden.float(), weight)
                    internal = torch.nn.functional.linear(hidden.float(), gate.weight_fp32)
                    external, _ = gate(hidden)
                    torch.npu.synchronize()
                    torch.testing.assert_close(internal, expected, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(external, expected, rtol=1e-5, atol=1e-5)
                print(f"PASS real Gate loader: enabled={enabled} bias={bias} internal_router={scheduled}", flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    print("PASS real Gate constructor/loading and FP32 logits; distributed MC2 and model accuracy not tested.")


if __name__ == "__main__":
    main()
