// Input checks provided by ACLNN on main, retained for the 0.26 custom launcher.
#pragma once
#include "mla_preprocess_validation.h"

namespace vllm_ascend {
inline void CheckDirectMlapoParameter(const c10::optional<at::Tensor> &tensor, const char *name,
                                    int64_t numel, at::ScalarType dtype, const at::Device &device)
{
    TORCH_CHECK(tensor.has_value(), "mla_preprocess_with_kv_offload: ", name, " is required");
    TORCH_CHECK(tensor->device() == device && tensor->scalar_type() == dtype &&
                tensor->numel() == numel && tensor->is_contiguous(),
                "mla_preprocess_with_kv_offload: ", name,
                " has invalid device, dtype, element count or strides");
}

inline void CheckDirectMlapoCompute(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, int64_t quantMode, int64_t ropeDim)
{
    const auto device = hiddenState.device();
    const auto dtype = hiddenState.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "Direct MLA requires FP16 or BF16 activations");
    TORCH_CHECK((quantMode == 0 || quantMode == 1) && wdqkv.scalar_type() == at::kChar &&
                wuq.scalar_type() == at::kChar,
                "Direct MLA requires quantized INT8 projection weights and quantization mode 0 or 1");
    TORCH_CHECK(quantMode != 1 || dtype == at::kBFloat16,
                "per_token_quant_symm only supports BF16 input");
    TORCH_CHECK(gamma1.dim() >= 1 && gamma2.dim() >= 1,
                "Direct MLA requires non-scalar norm weights");
    const int64_t n = hiddenState.size(0), d = hiddenState.size(1), q = gamma1.size(0);
    const int64_t h = wuk.size(0), nope = wuk.size(1), kv = wuk.size(2);
    const int64_t limit = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(n > 0 && n <= 1024 && d > 0 && d <= limit && q > 0 && q <= limit &&
                h > 0 && h <= limit && nope > 0 && nope <= limit && kv > 0 && kv <= limit &&
                ropeDim > 0 && ropeDim <= limit,
                "Direct MLA dimensions must be positive uint32 values and tokens in [1,1024]");
    TORCH_CHECK(q <= limit - kv - ropeDim && nope <= limit - ropeDim &&
                h <= limit / (nope + ropeDim) && h <= limit / kv && h <= limit / n,
                "Direct MLA dimensions exceed tiling limits");
    const int64_t downWidth = q + kv + ropeDim, upWidth = h * (nope + ropeDim);
    CheckDirectMlapoTensor(hiddenState, "hiddenState", {n, d}, dtype, device);
    CheckDirectMlapoTensor(wdqkv, "wdqkv", {1, (d + 31) / 32, downWidth, 32}, at::kChar, device);
    CheckDirectMlapoTensor(wuq, "wuq", {1, (q + 31) / 32, upWidth, 32}, at::kChar, device);
    CheckDirectMlapoTensor(wuk, "wuk", {h, nope, kv}, dtype, device);
    TORCH_CHECK(gamma1.device() == device && gamma1.scalar_type() == dtype &&
                gamma1.numel() == q && gamma1.is_contiguous() &&
                gamma2.device() == device && gamma2.scalar_type() == dtype &&
                gamma2.numel() == kv && gamma2.is_contiguous(),
                "Direct MLA norm weights have invalid device, dtype, element count or strides");
    CheckDirectMlapoParameter(beta1, "beta1", q, dtype, device);
    for (const auto *scale : {&descale0, &descale1}) {
        TORCH_CHECK(scale->has_value() &&
                    (scale->value().scalar_type() == at::kFloat || scale->value().scalar_type() == at::kLong),
                    "Direct MLA descale requires FP32 or INT64 metadata");
    }
    CheckDirectMlapoParameter(descale0, "descale0", downWidth, descale0->scalar_type(), device);
    CheckDirectMlapoParameter(descale1, "descale1", upWidth, descale1->scalar_type(), device);
    CheckDirectMlapoParameter(bias0, "bias0", downWidth, at::kInt, device);
    CheckDirectMlapoParameter(bias1, "bias1", upWidth, at::kInt, device);
    if (quantMode == 0) {
        CheckDirectMlapoParameter(quant_scale0, "quant_scale0", 1, dtype, device);
        CheckDirectMlapoParameter(quant_scale1, "quant_scale1", 1, dtype, device);
        CheckDirectMlapoParameter(quant_offset0, "quant_offset0", 1, at::kChar, device);
        CheckDirectMlapoParameter(quant_offset1, "quant_offset1", 1, at::kChar, device);
    }
    if (cos.has_value()) {
        CheckDirectMlapoParameter(cos, "cos", n * ropeDim, dtype, device);
        CheckDirectMlapoParameter(sin, "sin", n * ropeDim, dtype, device);
    }
}
}  // namespace vllm_ascend
