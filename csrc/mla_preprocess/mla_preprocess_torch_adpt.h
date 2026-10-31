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

// Python 调用经过本文件，再通过 ACLNN 启动底层 MlaPreprocess。
// 本文件有两个入口：mla_preprocess 写原分页缓存；mla_preprocess_with_kv_offload 写本批连续 KV。
// “是否额外输出 Q 中间结果”和“KV 用哪种存储布局”是两项设置，不能把 inner_out 当成 KV 的别名。
// 量化名称、RoPE 配对和维度告警共用 validation.h，实际计算继续交给底层算子。
#include "mla_preprocess_validation.h"

namespace vllm_ascend {
namespace mla_preprocess_detail {

inline int64_t ParseCacheMode(c10::optional<c10::string_view> cacheMode)
{
    const c10::string_view mode = cacheMode.value_or("krope_ctkv");
    if (mode == "kvcache") {
        return 0;
    }
    if (mode == "krope_ctkv") {
        return 1;
    }
    if (mode == "int8_nzcache") {
        return 2;
    }
    if (mode == "nzcache") {
        return 3;
    }
    TORCH_CHECK(false, "Unsupported cache_mode value: '", mode, "'");
    return 0;
}

// The kernel supports an enlarged dim0 stride for paged cache blocks. Every
// inner axis must still use the compact layout expected by the selected mode.
inline void ValidateCacheNonFirstAxisContiguous(const at::Tensor &cache, const char *tensorName)
{
    if (cache.dim() <= 1) {
        return;
    }
    const auto sizes = cache.sizes();
    const auto strides = cache.strides();
    int64_t expectedStride = 1;
    for (int64_t dim = cache.dim() - 1; dim >= 1; --dim) {
        if (sizes[dim] == 1) {
            continue;
        }
        TORCH_CHECK(strides[dim] == expectedStride,
                    tensorName, " dim", dim, " is non-contiguous: actual stride=", strides[dim],
                    ", expected contiguous stride=", expectedStride,
                    ". Only dim0/blockNum may be non-contiguous.");
        expectedStride *= sizes[dim];
    }
}

inline int64_t GetStride0(const at::Tensor &cache, const char *tensorName)
{
    TORCH_CHECK(cache.dim() >= 1, tensorName, " must have at least one dimension.");
    TORCH_CHECK(cache.stride(0) > 0, tensorName, " dim0 stride must be positive.");
    return cache.stride(0);
}

inline bool IsPhysicalNzCache(const at::Tensor &cache, int64_t cacheMode)
{
    return (cacheMode == 2 || cacheMode == 3) && cache.dim() == 4 && cache.size(2) != 1;
}

inline int64_t GetRopeHeadDim(const at::Tensor &kvCacheRope, int64_t cacheMode)
{
    TORCH_CHECK(kvCacheRope.dim() >= 1, "kv_cache_rope must not be a scalar.");
    if (IsPhysicalNzCache(kvCacheRope, cacheMode)) {
        return kvCacheRope.size(1) * kvCacheRope.size(3);
    }
    return kvCacheRope.size(-1);
}

}  // namespace mla_preprocess_detail

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope,
    const at::Tensor &slotmapping, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale, c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode, c10::optional<bool> enable_inner_out,
    at::Tensor &q_out0, at::Tensor &kv_cache_out0, at::Tensor &q_out1,
    at::Tensor &kv_cache_out1, at::Tensor &inner_out)
{
    mla_preprocess_detail::CheckRopePair(cos, sin);

    const int64_t cacheMode = mla_preprocess_detail::ParseCacheMode(cache_mode);
    const int64_t quantMode = mla_preprocess_detail::ParseQuantMode(quant_mode);
    const bool enableInnerOut = enable_inner_out.value_or(false);
    const bool enableRope = cos.has_value();
    // 原接口的 enable_inner_out 只决定是否额外导出 Q 的中间结果，默认 false；关闭它仍会计算最终 Q。
    // 原接口始终 directOutput=false：无论是否导出 inner_out，KV 都按 slot 写分页缓存。
    // ACLNN 新增的三个 CPU 写回参数也固定传 0，表示不在此接口中额外写映射 CPU 池。
    // 因此调用方可以“KV 仍写分页缓存，同时启用 enable_inner_out”，不必改用 Direct 接口。
    const bool directOutput = false;
    const int64_t noWriteback = 0;
    const at::Tensor slotMapping =
        slotmapping.scalar_type() == at::kLong ? slotmapping.to(at::kInt) : slotmapping;

    mla_preprocess_detail::ValidateCacheNonFirstAxisContiguous(kv_cache, "kv_cache");
    mla_preprocess_detail::ValidateCacheNonFirstAxisContiguous(kv_cache_rope, "kv_cache_rope");
    const int64_t kvCacheStride0 = mla_preprocess_detail::GetStride0(kv_cache, "kv_cache");
    const int64_t kvCacheRopeStride0 =
        mla_preprocess_detail::GetStride0(kv_cache_rope, "kv_cache_rope");

    const int64_t kvLoraRank = wuk.size(-1);
    const int64_t qkRopeHeadDim = mla_preprocess_detail::GetRopeHeadDim(kv_cache_rope, cacheMode);
    mla_preprocess_detail::WarnUnsupportedDimensions(kvLoraRank, qkRopeHeadDim);

    EXEC_NPU_CMD(
        aclnnMlaPreprocess,
        hiddenState,
        quant_scale0,
        quant_offset0,
        wdqkv,
        bias0,
        gamma1,
        beta1,
        quant_scale1,
        quant_offset1,
        gamma2,
        sin,
        cos,
        sin,
        cos,
        kv_cache,
        slotMapping,
        wuq,
        bias1,
        wuk,
        descale0,
        descale1,
        ctkv_scale,
        q_nope_scale,
        cacheMode,
        quantMode,
        enableInnerOut,
        enableRope,
        kvCacheStride0,
        kvCacheRopeStride0,
        directOutput,
        noWriteback,
        noWriteback,
        noWriteback,
        q_out0,
        kv_cache_out0,
        q_out1,
        kv_cache_out1,
        inner_out);

    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

// 新接口交付三类结果：最终 Q 的 q_out0/q_out1、Q 中间结果 inner_out、本批 KV 的 current_k/current_rope。
// inner_out 是归一化后、量化前的 Q 数据，不是当前 KV，也不是最终 Q 的第三个内容分片。
// 本接口固定 enableInnerOut=true、directOutput=true：既导出 Q 中间结果，又把 KV 按输入顺序连续写出。
// CPU 写回另外控制。若地址/容量为零，五份 NPU 输出照常生成；有效时，再将同一份 KV 写进 CPU 池。
// 例如 writeback_slots=[17,-1]：NPU current_k/current_rope 都写第 0/1 行，CPU 只更新第 17 行。
// 调用方先分配五个输出，本函数返回原张量的引用；捕获图会记住地址，重放时应更新内容而非随意换存储。
std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_with_kv_offload(
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
    // 先检查五个输出的形状、类型以及可选 CPU 地址范围，尽早发现调用错误。
    // 随后检查这些输出是否真的是可直接按行写入的 NPU 存储。
    // 这两层检查不同：一个张量在逻辑上 contiguous，不代表其 NPU 内部格式一定是普通 ND。
    const int64_t quantMode = CheckDirectMlapo(hiddenState, gamma1, cos, sin, wuk, quant_mode,
                                               q_out0, q_out1, inner_out, current_k, current_rope);
    CheckDirectMlapoWriteback(hiddenState, current_k, current_rope, writeback_slots,
                              writeback_k_base, writeback_rope_base, writeback_capacity);
    TORCH_CHECK(hiddenState.device().type() == c10::DeviceType::PrivateUse1,
                "mla_preprocess_with_kv_offload requires NPU tensors");
    // Compute inputs use the original operator's format handling. Direct
    // outputs are raw contiguous rows, so verify their physical layout here.
    // 计算输入继续允许原算子的权重布局；新输出必须是 ND/NCHW 兼容的连续存储。
    // 否则 Kernel 按连续行算出的地址，可能与特殊物理格式中实际的元素位置不一致。
    for (const at::Tensor *tensor : {&q_out0, &q_out1, &inner_out, &current_k, &current_rope}) {
        const auto format = at_npu::native::custom_ops::get_npu_format(*tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "mla_preprocess_with_kv_offload requires ND-compatible output storage");
    }
    if (writeback_slots.has_value()) {
        const auto format = at_npu::native::custom_ops::get_npu_format(writeback_slots.value());
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "MLAPO writeback_slots requires ND-compatible storage");
    }
    // Reuse main's ACLNN tiling/workspace lifecycle. The original operator
    // remains in its default paged-cache mode; this call selects Direct rows.
    // 共享 ACLNN 签名仍保留 slots 输入。开启 CPU 写回时，它逐行指定要更新 CPU 池的哪个 slot。
    // 写回关闭时，NPU current KV 只按行号寻址，不读取这些 slot 值；因此可以传一个占位张量。
    // 占位张量没有初始化并不影响这条关闭写回的路径，但不能把它拿来执行有效 CPU 写回。
    const at::Tensor slots = writeback_slots.has_value()
        ? writeback_slots.value()
        : at::empty({hiddenState.size(0)}, hiddenState.options().dtype(at::kInt));
    // 这三个参数分别回答三个问题：
    // cacheMode=1：K 和 K 的 RoPE 部分分成两份存储，对应 current_k/current_rope。
    // enableInnerOut=true：把已经算出的 Q 归一化结果另存到 inner_out，作为 q_c 交给 Indexer。
    // directOutput=true：第 i 个输入 token 的 KV 写到输出第 i 行，而不是按 slot 写分页 cache。
    // enableInnerOut 与 directOutput 不互相代替；只是这个 Direct 接口固定同时打开，未暴露任意组合。
    const int64_t cacheMode = 1;
    const bool enableInnerOut = true;
    const bool enableRope = cos.has_value();
    const bool directOutput = true;
    // 以下参数顺序必须与 ACLNN 定义保持一致。权重、scale 和 offset 直接沿用已有张量。
    // Direct 不通过重新标定量化参数来获得输出，也不为了返回 q_c 另算一遍归一化。
    // CPU 池地址和容量作为固定属性下传，每轮变化的 slots 则通过设备张量输入。
    const int64_t kvStride0 = current_k.stride(0);
    const int64_t ropeStride0 = current_rope.stride(0);
    EXEC_NPU_CMD(
        aclnnMlaPreprocess,
        hiddenState,
        quant_scale0,
        quant_offset0,
        wdqkv,
        bias0,
        gamma1,
        beta1,
        quant_scale1,
        quant_offset1,
        gamma2,
        sin,
        cos,
        sin,
        cos,
        current_k,
        slots,
        wuq,
        bias1,
        wuk,
        descale0,
        descale1,
        ctkv_scale,
        q_nope_scale,
        cacheMode,
        quantMode,
        enableInnerOut,
        enableRope,
        kvStride0,
        ropeStride0,
        directOutput,
        writeback_k_base,
        writeback_rope_base,
        writeback_capacity,
        q_out0,
        current_k,
        q_out1,
        current_rope,
        inner_out);
    return std::forward_as_tuple(q_out0, q_out1, inner_out, current_k, current_rope);
}

}  // namespace vllm_ascend

#endif  // MLA_PREPROCESS_TORCH_ADPT_H
