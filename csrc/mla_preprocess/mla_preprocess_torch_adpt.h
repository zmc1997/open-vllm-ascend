/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MLA_PREPROCESS_TORCH_ADPT_H
#define MLA_PREPROCESS_TORCH_ADPT_H


#include "op_host/mla_preprocess.h"
#include "mla_preprocess_direct_check.h"
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <initializer_list>

namespace vllm_ascend {
// Preserve the existing launcher ABI while selecting a separately compiled kernel.
decltype(mla_preprocess_impl) mla_preprocess_direct_impl;

constexpr int64_t MLA_PREPROCESS_FULLY_SUPPORTED_KV_LORA_RANK = 512;
constexpr int64_t MLA_PREPROCESS_FULLY_SUPPORTED_QK_ROPE_HEAD_DIM = 64;

template <bool DirectOutput>
std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_dispatch(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor *slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode,
    c10::optional<bool> enable_inner_out, at::Tensor &q_out0,
    at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1, at::Tensor &inner_out,
    int64_t writeback_k_base = 0, int64_t writeback_rope_base = 0, int64_t writeback_capacity = 0)
{
    const c10_npu::OptionalNPUGuard npu_guard(hiddenState.device());
    at::Tensor Descale0 =
        descale0.has_value()
            ? descale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Descale1 =
        descale1.has_value()
            ? descale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Beta1 =
        beta1.has_value()
            ? beta1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale0 =
        quant_scale0.has_value()
            ? quant_scale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale1 =
        quant_scale1.has_value()
            ? quant_scale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset0 =
        quant_offset0.has_value()
            ? quant_offset0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset1 =
        quant_offset1.has_value()
            ? quant_offset1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias0 =
        bias0.has_value()
            ? bias0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias1 =
        bias1.has_value()
            ? bias1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    bool enableInnerOut =
        enable_inner_out.has_value()
            ? enable_inner_out.value()
            : false;
    TORCH_CHECK(
        cos.has_value() == sin.has_value(),
        "mla_preprocess requires cos and sin to both be tensors or both be None.");
    bool enableRope = cos.has_value();
    // Use placeholder tensors because device kernels initialize RoPE input addresses unconditionally.
    at::Tensor Cos =
        cos.has_value()
            ? cos.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Sin =
        sin.has_value()
            ? sin.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    int64_t kvLoraRank = wuk.size(-1);
    int64_t qkRopeHeadDim = kv_cache_rope.size(-1);
    if (kvLoraRank != MLA_PREPROCESS_FULLY_SUPPORTED_KV_LORA_RANK ||
        qkRopeHeadDim != MLA_PREPROCESS_FULLY_SUPPORTED_QK_ROPE_HEAD_DIM) {
        TORCH_WARN_ONCE(
            "mla_preprocess currently fully supports only kv_lora_rank=",
            MLA_PREPROCESS_FULLY_SUPPORTED_KV_LORA_RANK,
            " and qk_rope_head_dim=",
            MLA_PREPROCESS_FULLY_SUPPORTED_QK_ROPE_HEAD_DIM,
            ", but received kv_lora_rank=",
            kvLoraRank,
            " and qk_rope_head_dim=",
            qkRopeHeadDim,
             ". Inputs outside the fully supported configuration are allowed to continue, "
             "but may produce accuracy issues.");
    }
    auto [workspace_tensor, tiling_buffers, block_dim] = mlapo::mla_preprocess_tiling(
        hiddenState,
        wdqkv,
        wuk,
        gamma1,
        kv_cache,
        kv_cache_rope,
        cache_mode,
        quant_mode,
        enableInnerOut,
        enableRope,
        DirectOutput,
        writeback_k_base,
        writeback_rope_base,
        writeback_capacity
    );

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = Quant_scale0.data_ptr();
    void *quant_offset0_ptr = Quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = Bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = Beta1.data_ptr();
    void *quant_scale1_ptr = Quant_scale1.data_ptr();
    void *quant_offset1_ptr = Quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *sin_ptr = Sin.data_ptr();
    void *cos_ptr = Cos.data_ptr();
    void *kv_cache_ptr = kv_cache.data_ptr();
    void *slotmapping_ptr = nullptr;
    if (slotmapping != nullptr) {
        slotmapping_ptr = slotmapping->data_ptr();
    }
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = Bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = Descale0.data_ptr();
    void *descale1_ptr = Descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *kv_cache_out0_ptr = kv_cache_out0.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *kv_cache_out1_ptr = kv_cache_out1.data_ptr();
    void *inner_out_ptr = inner_out.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling_buffers->device.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name(DirectOutput ? "mla_preprocess_without_kv_cache_write" : "mla_preprocess");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                          gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr,
                          kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                          qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                          tiling_buffers, tiling_ptr, block_dim]() -> int {
        mlapo::InitializeMlaTiling(tiling_buffers, stream);
        if constexpr (DirectOutput) {
            mla_preprocess_direct_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                                gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr,
                                kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                                qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                                tiling_ptr, block_dim);
        } else {
            mla_preprocess_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                                gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr,
                                kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                                qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                                tiling_ptr, block_dim);
        }
        return 0;
    });
    cmd.Run();
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode,
    c10::optional<bool> enable_inner_out, at::Tensor &q_out0,
    at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1, at::Tensor &inner_out)
{
    return mla_preprocess_dispatch<false>(
        hiddenState, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2, cos, sin,
        wuk, kv_cache, kv_cache_rope, &slotmapping, quant_scale0, quant_offset0, bias0,
        quant_scale1, quant_offset1, bias1, ctkv_scale, q_nope_scale, cache_mode, quant_mode,
        enable_inner_out, q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>
mla_preprocess_without_kv_cache_write(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale, c10::optional<c10::string_view> quant_mode,
    at::Tensor &q_out0, at::Tensor &q_out1, at::Tensor &inner_out,
    at::Tensor &current_k, at::Tensor &current_rope,
    const c10::optional<at::Tensor> &writeback_slots,
    int64_t writeback_k_base, int64_t writeback_rope_base, int64_t writeback_capacity)
{
    CheckDirectMlapo(hiddenState, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2,
                     cos, sin, wuk, quant_scale0, quant_offset0, bias0, quant_scale1,
                     quant_offset1, bias1, quant_mode, q_out0, q_out1, inner_out,
                     current_k, current_rope);
    CheckDirectMlapoWriteback(hiddenState, writeback_slots, writeback_k_base,
                             writeback_rope_base, writeback_capacity);
    TORCH_CHECK(hiddenState.device().type() == c10::DeviceType::PrivateUse1,
                "mla_preprocess_without_kv_cache_write requires NPU tensors");
    // Each scale is a scalar view into an expanded copy of the original MLAPO scale.
    // The existing kernel copy reads a full block, so validate physical backing.
    for (const at::Tensor *scale : {&quant_scale0.value(), &quant_scale1.value()}) {
        const auto offset = scale->storage_offset() * scale->element_size();
        const auto bytes = scale->storage().nbytes();
        TORCH_CHECK(bytes >= offset && bytes - offset >= 32,
                    "mla_preprocess_without_kv_cache_write: scale view requires at least "
                    "32 readable storage bytes; pass a view of the expanded original scale buffer");
    }
    // Contiguous logical strides alone do not prove the storage has ND layout.
    for (const at::Tensor *tensor : std::initializer_list<const at::Tensor *>{&hiddenState, &wuk, &gamma1, &gamma2,
             &beta1.value(), &descale0.value(), &descale1.value(), &cos.value(), &sin.value(),
             &quant_scale0.value(), &quant_scale1.value(), &quant_offset0.value(), &quant_offset1.value(),
             &bias0.value(), &bias1.value(), &q_out0, &q_out1, &inner_out, &current_k, &current_rope}) {
        const auto format = at_npu::native::get_npu_format(*tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "mla_preprocess_without_kv_cache_write requires ND-compatible input and output storage");
    }
    TORCH_CHECK(at_npu::native::get_npu_format(wdqkv) == ACL_FORMAT_FRACTAL_NZ &&
                at_npu::native::get_npu_format(wuq) == ACL_FORMAT_FRACTAL_NZ,
                "mla_preprocess_without_kv_cache_write requires transformed NZ projection weights");
    if (writeback_slots.has_value()) {
        const auto format = at_npu::native::get_npu_format(writeback_slots.value());
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "MLAPO writeback_slots requires ND-compatible storage");
    }
    // Reuse the arithmetic and graph-safe launch. Direct tiling uses only the
    // output descriptors; optional business slots address only mapped CPU pools.
    mla_preprocess_dispatch<true>(
        hiddenState, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2, cos, sin,
        wuk, current_k, current_rope, writeback_slots.has_value() ? &writeback_slots.value() : nullptr,
        quant_scale0, quant_offset0, bias0,
        quant_scale1, quant_offset1, bias1, ctkv_scale, q_nope_scale,
        c10::string_view("krope_ctkv"), quant_mode, true,
        q_out0, current_k, q_out1, current_rope, inner_out,
        writeback_k_base, writeback_rope_base, writeback_capacity);
    return std::forward_as_tuple(q_out0, q_out1, inner_out, current_k, current_rope);
}
}
#endif
