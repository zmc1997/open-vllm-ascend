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
