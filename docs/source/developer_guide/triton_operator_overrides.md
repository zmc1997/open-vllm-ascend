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
