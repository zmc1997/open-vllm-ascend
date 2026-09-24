# Fused MC2 shared expert scheduling: scheme 3

This experimental branch targets GLM-5.2 (`GlmMoeDsaForCausalLM`) with integer
W8A8/W4A8 routed experts and INT8 shared projections. Enable explicitly:

```bash
vllm serve "$MODEL" --additional-config '{"enable_fused_mc2":1,"fused_mc2_shared_schedule":true}'
```

Retain the deployment's existing TP/EP, dtype, quantization and model arguments.
The default is `false`; the existing `multistream_overlap_shared_expert` option
remains incompatible with fused MC2. Restart after changing the option: GLM's
bias-free gate prepares its FP32 weight during weight loading when opted in.

## Dependencies

The main stream runs gating, TopK, `dispatch_ffn_combine`, and routed finalization.
The shared stream runs quant, Gateup, SwiGLU, and Down. Device events enforce:

| Shared stage | Additional dependency |
| --- | --- |
| quant | Shared input ready; may overlap gating |
| Gateup | Gating complete; may overlap TopK |
| SwiGLU | Routing selection complete; may overlap the fused operator |
| Down | Routed forward complete, including finalization/TP reduction |

The Down wait is intentionally stronger than the original drawing: it includes
routed finalization, because the completion event is recorded at the runner's
return boundary. Shared TP reduction is preserved after joining the shared stream.
This may delay Down further than waiting at the custom kernel boundary alone.
No fixed delays, host synchronization, or modification of communication singletons
is used. Events order launches but do not guarantee simultaneous kernel residency.

FlashComm input gathering runs on the main stream before the shared input event;
the router retains its original token layout. Cross-stream input/output lifetimes
are recorded with `record_stream`; graph capture/replay still requires NPU validation.

## Supported and fallback paths

Scheduling requires the actual per-forward `FUSED_MC2` path with
`enable_fused_mc2=1`, no MegaMoE, an internal router, routed `silu`, and the stock
`AscendW8A8DynamicFusedMoEMethod` or `AscendW4A8DynamicFusedMoEMethod`. Shared
projections must have INT8 weights, scales, no bias, and no separate expert gate.
SiTU, active LoRA and mixed shared/routed expert placement are excluded. Incompatible forwards retain baseline execution
and emit a once-per-reason `scheme 3 fallback` diagnostic. Prefill may select a
different communication method and fall back even when decode is eligible.

The staged shared operations preserve baseline integer math and activation scale,
clamp, alpha and beta handling (including W4A8's default alpha/beta). No vLLM source
changes are required. GLM without shared experts retains its external router.

## Validation

CPU checks execute the scheduling dependency graph, the real shared math method
with fake operators, callback forwarding through the runner and quantization
adapter, and the GLM gate weight-loading contract:

```bash
python tests/ut/ops/test_shared_schedule_standalone.py -v
python tests/ut/ops/test_glm_gate_schedule_standalone.py -v
```

Set `ASCEND_SOURCE_ROOT` when invoking a copied test outside its source checkout.

On an Ascend host with the branch installed, the following imports the deployed
package and exercises real integer shared kernels, cross-stream output joins, and
changed-input graph replay. It raises an error when NPU support is absent:

```bash
python tests/e2e/nightly/single_node/ops/singlecard_ops/verify_shared_schedule3.py
```

The smoke compares sequential and scheduled execution of the same shared kernels;
it validates scheduling consistency, not independent numerical kernel correctness.
It uses a synthetic routed calculation and **does not validate EP or
the fused dispatch kernel**. It has not been run on this Windows development host.
Before use, compare the same GLM-5.2 checkpoint with this option off/on, identical
TP/EP and requests, both eager and graph modes, decode token counts including 6,
prefill and empty/padded ranks. Compare outputs/logprobs within the deployment's
numerical tolerance, ensure no fallback during the intended decode path, and
profile the actual dependencies and entire MoE layer latency. Measure Down and
fused-op duration as well as total latency; no performance gain is claimed yet.
