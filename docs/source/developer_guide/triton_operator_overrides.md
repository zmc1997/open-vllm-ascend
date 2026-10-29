# Experimental Triton operator overrides

These opt-in implementations start from the Layerwise Direct baseline
`2cc4f2772b46520ab475bc47940b4c529385de1f`. They preserve the surrounding
parallel strategy and the original projection GEMM. All switches default to
`false`; an enabled implementation does not silently fall back to CANN.

## o_proj input static quantization

Set `enable_oproj_triton_quant` in `additional_config` to replace the input
static quantization of linear layers whose prefix ends in `.o_proj`.
Already-INT8 inputs still bypass quantization. The original reciprocal and
offset tensors remain the parameter owners, including their dtype and live
updates during graph replay. GEMM, bias, dequantization and reduction remain
unchanged.

`static_quantize.quantize` accepts BF16, FP16 and FP32 ND inputs and returns
contiguous INT8 with the original shape. Its tiling uses actual dimensions,
strides, dtype sizes and CANN device resources. It has no batch-size, TP-size
or main-model/MTP whitelist. The numerical contract retains multiplication,
addition, saturation and round-to-nearest-even without FP fusion.

CPU checks, run from this checkout:

```bash
python tests/ut/quantization/test_oproj_triton_quant.py --source-root .
python tests/ut/ops/test_static_quantize_tiling.py
python tests/ut/ops/test_kernel_resources.py --source vllm_ascend/ops/triton/kernel_resources.py
```

The device gate is `tests/ut/ops/test_static_quantize.py`. It requires an
isolated NPU and captured real parameter cases via `--parameter-json`; see
`--help`. Install this checkout so the gate imports the matching resource
module. Checks compare output bytes, input immutability and live graph inputs.

For continuous per-channel inputs with fewer rows than hardware cores, the
planner derives a row-piece size from the dtype, UB budget and alignment,
then avoids creating more cores than those pieces require. This is a Triton
scheduling choice; it does not assume that CANN's core count produces the
same physical execution cost. Original scalar, strided-parameter and large-row
plans are retained. A transposed input normalized by the existing copy path
can also use the continuous-input plan.

On Ascend 910_9382 with CANN 9.1, TEST-065 passed 1,842 strict checks over
101 captured parameter sets and 504 checks in the 42-case full-call matrix.
For BF16 `[6,8192]`, CANN / preceding Triton / this candidate measured
6.275 / 4.178 / 2.374 us. FP32 `[512,8192]` remains slower than CANN
(18.125 vs 16.875 us); BF16 at that size also measured slower in this batch
(16.833 vs 16.375 us). These are isolated operator measurements, not model
ITL/TPOT or a speedup promise for every shape, device and compiler version.

## Independent V-up BMM

Set `enable_vup_triton_bmm` in `additional_config` to replace only the V-up
batch matrix multiplication. It is independent of static quantization and
does not replace the o_proj INT8 GEMM. Dimensions and strides come from the
actual input and weight tensors; contiguous operands avoid normalization
copies. Necessary copies for noncontiguous inputs are included in full-call
performance measurements.

FP32 mode 3 preserves the original HF32 behavior through a small AscendC
control helper linked into the Triton kernel. The adjacent
`vup_hf32_control.bc` and its source are package data; this is a Triton BMM
with a control helper, not an entirely Triton-only implementation.

```bash
python tests/ut/attention/test_vup_triton_bmm.py --source-root .
python tests/ut/ops/test_vup_bmm.py --source-root .
python tests/ut/ops/test_vup_bmm_packaging.py
python tests/ut/ops/test_vup_bmm_npu.py --self-test
```

The device driver is `test_vup_bmm_npu.py`; use `--help` for the isolated
NPU and source/hash arguments. TEST-063 completed 819 ordinary and 453 mode-3
checks, including control restoration, and a separate corrected raw-copy gate.
All 30 full-call performance case means improved on CANN. Those results
cover the recorded dtype/shape/layout matrix on Ascend 910_9382/CANN 9.1;
full-model validation of the current implementation is a separate gate.
Publication cleanup also removes an unreachable FP32 branch from the
half-precision kernel. FP32 retains its separate kernel and HF32 controls;
the active half-precision arithmetic is unchanged. CPU dispatch and kernel
checks cover this cleanup. TEST-064 subsequently passed 576 byte checks for
the BMM -> original reshape -> quantization chain, including live graph
updates; FP32 mode-0 chain regressions were retained in the report. TEST-065
then passed all 24 combined-chain cases with the new quantization schedule;
BF16 main/MTP-shaped chains measured 9.477/9.447 us vs CANN 18.236/18.133 us.
These inputs are synthetic with captured parameter values, not captured model
activations; the chain excludes GEMM and reduction.

## Indexer Q static quantization

Set `enable_indexer_q_triton_quant` in `additional_config` to select the
same static quantization implementation for `.indexer.wq_b` linear layers.
The switch is independent of the o_proj and BMM switches. This changes the
static quantization before the Q projection, not Q dynamic quantization.
There is no duplicate kernel and no special batch-size or width restriction.

```bash
python tests/ut/quantization/test_indexer_q_triton_quant.py --source-root .
```

TEST-065 measured BF16 `[6,2048]` at 1.500 us for this candidate,
4.212 us for the preceding Triton implementation and 7.020 us for CANN,
using the same inputs and parameter representation in one batch.
The single-operator saving must not be presented as model latency improvement.

## Indexer K Cache writes

Set `enable_indexer_k_triton_cache` in `additional_config` to replace the
original scatter updates for K data and scale. These remain two separate
calls, sharing one implementation. The existing `store_kv_block` route keeps
its priority; original notifications remain after the writes.

Dimensions, source/cache strides and element sizes come from the tensors.
Writes preserve raw bits and the original cache alias. For repeated valid
slots, the last input update wins; `-1` padding slots are skipped. Behavior
for arbitrary invalid indices has not been established against CANN and
must not be claimed as fully equivalent. Disjoint cache-address ranges belong
to different owners; each owner scans the input in order. This avoids sorting,
workspace and a cross-core barrier. Nearby slots may serialize on one owner;
this is a documented limitation, not a performance guarantee for every slot
distribution.

```bash
python tests/ut/attention/test_indexer_k_triton_cache.py --source-root .
python tests/ut/ops/test_indexer_cache.py --source-root .
python tests/ut/ops/test_indexer_cache_npu.py --self-test
```

TEST-064 completed 328 same-layout CANN cases and 72 reference-only cases
for CANN-rejected output layouts with the retained copy05 implementation;
all recorded checks passed. TEST-065 also compared it with CANN and an experimental cyclic-owner variant in one
62-case, 1,060-check batch. At the recorded production cache capacity, K writes
measured 4.513 vs CANN 6.953 us, and scale writes 4.556 vs CANN 6.959 us.
The cyclic-owner variant passed correctness but added almost no benefit for
those inputs and regressed dispersed-slot cases by 6.359-8.481 us relative to
copy05, so it is not included. Binary-matched physical IR for the recorded
copy05 signatures confirms direct source-to-cache copies; this does not prove
all possible compiler signatures or remove the owner-load limitation.

## Same-batch operator comparisons

After the strict gates pass, the full-call layout benchmark can compare
each implementation with CANN and optionally a frozen previous source:

```bash
python tests/ut/ops/benchmark_operator_layouts.py \
  --source-dir vllm_ascend/ops/triton --kind quant --kind bmm --kind cache \
  --dtype bf16 --dtype fp16 --dtype fp32 --device 0 --output result.json
```

Run only on an idle, isolated NPU. The tool checks bytes, graph input
freshness, input immutability and Cache aliases before timing. Event times
include internal copies and are averaged over graph repetitions; they are
not individual kernel durations or model ITL/TPOT. The shapes and slot
distributions in this benchmark are fixtures, not production restrictions.

These four overrides do not include the discarded LayerNorm, Q dynamic
quantization, independent TP4 or replacement INT8 GEMM experiments. Model
testing remains a separate step after reviewing the operator results.
