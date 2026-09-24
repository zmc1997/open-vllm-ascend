# Fused MC2 shared expert schedule: scheme 1

This experimental branch targets GLM-5.2 (`GlmMoeDsaForCausalLM`, the
DeepseekV2 MoE implementation). It is disabled by default. Enable with:

```text
--additional-config '{"enable_fused_mc2": 1, "fused_mc2_shared_schedule": true}'
```

Keep model, quantization, TP/EP and graph settings identical across comparisons.
The option preserves the existing prohibition on enabling
`multistream_overlap_shared_expert` with fused MC2.

## Dependencies

Shared Quant -> Gateup runs first. Gating waits for Gateup; shared SwiGLU can
run alongside Gating. Shared Down waits for Gating and can run alongside TopK.
The callback after TopK waits for Down before DispatchFFNCombine. Final shared
reduction remains on the original stream. No fixed delays or CPU synchronization
are introduced. Cross-stream inputs/outputs register allocator stream usage.
Shared input preparation stays on the main stream to preserve collective order.

## Eligibility and fallback

Requires the actual FUSED_MC2 context with `enable_fused_mc2=1`, no MegaMoE,
an internal FP32 router, standard SiLU routed activation, integer W8A8/W4A8
routed quantization, and bias-free int8 shared weights with per-channel scales.
W4A8 refers to routed experts; shared int4/grouped-weight layouts are unsupported.
SiTU, shared expert gates, active LoRA, external routers, MXFP/FP8 and non-FUSED
contexts fall back with a once-only diagnostic. GLM's gate is pre-cast only when
the new option is enabled; layers without shared experts retain external routing.

Shared math preserves the baseline scales, clamp and activation parameters.
W4A8 retains the baseline default alpha/beta. Optional `None` parameters normalize
exactly as the runtime input builder does.

## Validation

Run from this checkout (Python 3.10+; CPU tests need only the standard library):

```bash
python tests/ut/ops/test_fused_mc2_scheme1_standalone.py --source-root . -v
python tests/ut/ops/test_glm_gate_schedule_standalone.py -v
```

Standalone tests execute actual runner, adapter and quantization method bodies
with scalar operator/stream doubles. They check event dependencies, stream lifetime,
math against the original shared path, fallback and callback placement.

On an NPU host with the patched package installed:

```bash
python tests/ut/ops/test_fused_mc2_shared_npu_smoke.py --tokens 6 --hidden 6144
```

This smoke uses actual shared NPU operators, the actual scheduler and real TopK;
it checks both integer routed mode parameter sets against the baseline. Routed
computation is a stub: this is not a distributed MC2 or model accuracy test.
Missing NPU/dependencies is an error. Local development has no NPU, so this test
is pending. Full GLM-5.2 validation must check baseline vs patched token/logit
accuracy, eager and captured execution, multi-rank collective progress, memory
stability, and end-to-end layer latency with profiling. No speedup is claimed.
