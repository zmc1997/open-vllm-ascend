# Fused MC2 shared expert schedule: scheme 2

This experimental branch targets GLM5.2 using the `GlmMoeDsaForCausalLM`
architecture / DeepseekV2 MoE runner. Enable with
`--additional-config '{"enable_fused_mc2":1,"fused_mc2_shared_schedule":true}'`.
The new boolean defaults to false. The existing
`multistream_overlap_shared_expert` incompatibility remains unchanged.

The schedule is: shared quant → Gateup; then gating and shared SwiGLU may
execute concurrently. Shared Down waits for gating, while the main stream
runs TopK → DispatchFFNCombine without waiting for Down. The main stream
joins the shared stream before shared output reduction/consumption.
Actual resource overlap is hardware dependent; no timing delays are used.

Only actual FUSED_MC2 batches using DispatchFFNCombine qualify. MegaMoE,
external routers, mix placement, SiTU/non-SiLU routed activation, shared
expert gates, LoRA, biased projections and other quantization methods fall
back with a once-only diagnostic. Routed integer W8A8/W4A8 is supported;
shared projections must have int8 weights and the existing per-channel
scale tensors. FP8/MXFP and packed int4 shared weights are unsupported.
The GLM gate prepares its FP32 weight with the opt-in setting so that the
runner can calculate gating internally. Layers without shared experts
retain their external routing path.

FlashComm input preparation runs on the main stream before the fork. The
router consumes the original token shard. Shared TP/DP output finalization
runs on main after both branches finish, retaining existing collective
order and reduction semantics. Cross-stream inputs and outputs are recorded
with the allocator; all event waits follow their event records. Graph capture
and replay on the deployed torch_npu/CANN version still require NPU validation.

## Validation

From the source root, with Python 3.11+:

```bash
python tests/ut/ops/test_fused_mc2_schedule_standalone.py -v
python tests/ut/ops/test_glm_gate_schedule_standalone.py -v
```

The tests execute extracted production methods against a fake stream DAG;
they check dependencies, absence of a Down → Dispatch edge, final join,
activation metadata, runner dispatch, fallback and gate handling. Set
`ASCEND_SOURCE_ROOT` to the source/site-packages directory when copying the
tests elsewhere. They do not validate NPU kernels or performance.

A single-device smoke test needs no model factory and exercises actual shared
quantization/matmul/activation plus the production scheduler:

```bash
python tests/ut/ops/test_fused_mc2_shared_npu_smoke.py --tokens 6 --hidden 6144 --iterations 5
```

It compares bitwise output against the original shared path for W8A8/W4A8
routed metadata. Routing uses a single-device TopK stub; it does not test
distributed DispatchFFNCombine. Missing NPU/dependencies fail rather than skip.

Multi-rank DispatchFFNCombine, graph capture/replay and full-model accuracy
remain Pod validation requirements. Compare the same checkpoint, quantization,
TP/EP, prompts, batch shape and warmup with the option false/true. Inspect
profiler stream ordering and full-layer latency, including collectives.
No NPU correctness or performance result has been obtained locally.
