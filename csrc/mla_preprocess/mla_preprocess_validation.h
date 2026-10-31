// Shared MLAPO configuration and Direct output/writeback checks.
// 这个文件检查五个 NPU 输出张量是否准备正确，以及可选的 CPU 写回地址是否成套有效。
// 五个输出不是五份 KV：q_out0/q_out1 是最终 Q 的两个部分，inner_out 是 Q 的中间结果，
// current_k/current_rope 才是本批新 KV。CPU 写回只使用 current_k/current_rope 对应的数据。
// 真实 NPU 调用和编译时的 Meta 调用共用这些检查；Meta 只看形状、类型等信息，不执行计算。
// 权重格式等原有计算输入仍由 ACLNN/Host 检查，不能把本文件看成整个算子的全部校验。
#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <limits>

namespace vllm_ascend {
namespace mla_preprocess_detail {
constexpr int64_t FULLY_SUPPORTED_KV_LORA_RANK = 512;
constexpr int64_t FULLY_SUPPORTED_QK_ROPE_HEAD_DIM = 64;

inline int64_t ParseQuantMode(c10::optional<c10::string_view> quantMode)
{
    // quantMode 决定采用哪一种量化计算，不决定 Q 中间结果是否导出，也不决定 KV 写到哪里。
    // Python 传入名称，Host/Kernel 使用整数：per_tensor_quant_asymm 对应 0，per_token_quant_symm 对应 1。
    // 不传名称时沿用原默认值 per_token_quant_symm；拼错或未支持的名称会立即报错。
    // 名称解析通过后，还要检查 dtype、权重和输出模式是否支持这个组合。
    const c10::string_view mode = quantMode.value_or("per_token_quant_symm");
    if (mode == "per_tensor_quant_asymm") {
        return 0;
    }
    if (mode == "per_token_quant_symm") {
        return 1;
    }
    if (mode == "per_token_quant_asymm") {
        return 2;
    }
    if (mode == "no_quant") {
        return 3;
    }
    TORCH_CHECK(false, "Unsupported quant_mode value: '", mode, "'");
    return 0;
}

inline void CheckRopePair(const c10::optional<at::Tensor> &cos,
                          const c10::optional<at::Tensor> &sin)
{
    // sin/cos 是 RoPE 位置旋转需要的两份系数。两份都提供时，后续入口启用 RoPE；
    // 两份都不提供时，关闭位置旋转，但矩阵计算和 Q/KV 输出仍照常进行。
    // 只给 sin 或只给 cos 不能完成旋转，因此报错，避免把残缺输入误当作“关闭 RoPE”。
    TORCH_CHECK(cos.has_value() == sin.has_value(),
                "mla_preprocess requires cos and sin to both be tensors or both be None.");
}

inline void WarnUnsupportedDimensions(int64_t kvLoraRank, int64_t qkRopeHeadDim)
{
    // kvLoraRank 是压缩 K 每行的元素数，qkRopeHeadDim 是位置编码部分每行的元素数。
    // 完整支持的默认配置是 512/64；偏离默认值时只告警，保留原算子允许继续执行的行为。
    // 例如传入 256/64 不会单因这项检查被拒绝，但仍需验证输出数值，不能用“没报错”代替验证。
    if (kvLoraRank != FULLY_SUPPORTED_KV_LORA_RANK ||
        qkRopeHeadDim != FULLY_SUPPORTED_QK_ROPE_HEAD_DIM) {
        TORCH_WARN_ONCE(
            "mla_preprocess currently fully supports only kv_lora_rank=",
            FULLY_SUPPORTED_KV_LORA_RANK,
            " and qk_rope_head_dim=", FULLY_SUPPORTED_QK_ROPE_HEAD_DIM,
            ", but received kv_lora_rank=", kvLoraRank,
            " and qk_rope_head_dim=", qkRopeHeadDim,
            ". Inputs outside the fully supported configuration are allowed to continue, "
            "but may produce accuracy issues.");
    }
}
}  // namespace mla_preprocess_detail

inline void CheckDirectMlapoTensor(const at::Tensor &tensor, const char *name,
                                   at::IntArrayRef shape, at::ScalarType dtype,
                                   const at::Device &device)
{
    // tensor 是调用方交给 Kernel 写入的输出空间，shape/dtype/device 是该输出必须满足的约定。
    // Direct 用“行号 * 行长度”算写入位置，因此输出必须连续；转置视图即使元素总数正确也不能直接用。
    // 检查失败就报错，不会偷偷申请另一块输出或改变返回张量。实际 NPU 入口还会检查物理格式。
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype &&
                tensor.sizes() == shape && tensor.is_contiguous(),
                "mla_preprocess_with_kv_offload: ", name,
                " must be contiguous on ", device, " with dtype ", dtype,
                " and shape ", shape, "; got ", tensor.device(), ", ", tensor.scalar_type(),
                ", ", tensor.sizes());
}

inline void CheckDirectMlapoWriteback(
    const at::Tensor &hiddenState, const at::Tensor &current_k, const at::Tensor &current_rope,
    const c10::optional<at::Tensor> &slots, int64_t k_base, int64_t rope_base, int64_t capacity)
{
    // k_base/rope_base/capacity 全为 0：只生成 NPU 上的 Q、inner_out 和当前 KV，不更新 CPU 池。
    // CPU 写回开启：两个地址和容量都必须有效，slots 中每个整数指定对应 token 更新 CPU 池的哪一行。
    // 例如 slots=[17,-1]，NPU 仍生成两行 KV；开启写回时只更新 CPU 第 17 行，-1 行的 CPU 写回跳过。
    // CPU 写回关闭不等于关闭 Direct，更不等于关闭 Q/KV 计算；其他 TP rank 正是用零值避免重复写池。
    if (k_base == 0 && rope_base == 0 && capacity == 0) {
        return;
    }
    TORCH_CHECK(slots.has_value(), "MLAPO writeback addresses/capacity require writeback_slots");
    CheckDirectMlapoTensor(slots.value(), "writeback_slots", {hiddenState.size(0)},
                           at::kInt, hiddenState.device());
    TORCH_CHECK(k_base > 0 && rope_base > 0 && capacity > 0,
                "MLAPO writeback requires positive mapped CPU bases and capacity");
    TORCH_CHECK(k_base % 32 == 0 && rope_base % 32 == 0,
                "MLAPO writeback mapped CPU bases must be 32-byte aligned");
    // 写回目的位置 = 池起始地址 + slot * 每行字节数。capacity 表示能放多少行，不是字节数。
    // 在计算完整地址范围前先检查乘加是否溢出，并拒绝 K 池和 RoPE 池占用重叠内存。
    // 这里只拿到了整数地址，无法判断地址背后的 CPU 内存是否真的已注册、映射并仍然存活；
    // 调用方必须保证这些条件，不能把任意普通 CPU 张量的地址当成可写回的池。
    const int64_t address_limit = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(current_k.size(1) <= address_limit / current_k.element_size() &&
                current_rope.size(1) <= address_limit / current_rope.element_size(),
                "MLAPO writeback row byte count overflows int64");
    const int64_t k_bytes = current_k.size(1) * current_k.element_size();
    const int64_t rope_bytes = current_rope.size(1) * current_rope.element_size();
    TORCH_CHECK(capacity <= (address_limit - k_base) / k_bytes &&
                capacity <= (address_limit - rope_base) / rope_bytes,
                "MLAPO writeback mapped CPU address range overflows int64");
    TORCH_CHECK(k_base + capacity * k_bytes <= rope_base ||
                rope_base + capacity * rope_bytes <= k_base,
                "MLAPO writeback mapped CPU K and RoPE ranges must not overlap");
}

inline int64_t CheckDirectMlapo(
    const at::Tensor &hiddenState, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, c10::optional<c10::string_view> quant_mode,
    const at::Tensor &q_out0, const at::Tensor &q_out1, const at::Tensor &inner_out,
    const at::Tensor &current_k, const at::Tensor &current_rope)
{
    // Common compute inputs retain the original ACLNN/Host checks. Only the
    // new contiguous outputs and CPU destinations need Direct-specific checks.
    mla_preprocess_detail::CheckRopePair(cos, sin);
    TORCH_CHECK(hiddenState.dim() == 2, "mla_preprocess_with_kv_offload: hiddenState must be [N,D]");
    const auto device = hiddenState.device();
    const auto dtype = hiddenState.scalar_type();
    TORCH_CHECK(wuk.dim() == 3 && gamma1.dim() >= 1 && current_rope.dim() == 2,
                "mla_preprocess requires rank-3 wuk, non-scalar gamma1 and rank-2 current_rope");
    // N 是输入 token 数，H 是本 rank 的注意力头数。下面逐一检查五个输出空间：
    // q_out0 [N,H,kv_rank] 和 q_out1 [N,H,rope_dim]：最终 Q 的内容部分和位置编码部分，供注意力使用。
    // inner_out [N,q_rank]：Q 在归一化后、量化前的中间结果，当前接入称为 q_c，供 Indexer 使用。
    // current_k [N,kv_rank] 和 current_rope [N,rope_dim]：本批 K 和 K 的位置编码部分，供 KV 消费路径使用。
    // inner_out 不是 KV，导出 inner_out 不会改变 KV 存储位置；Direct 入口在这里固定要求五份输出都提供。
    // current_k/current_rope 按输入第 0、1、2 行排列；CPU 池位置由 slots 另外指定。
    const int64_t n = hiddenState.size(0);
    const int64_t h = wuk.size(0);
    const int64_t kv_rank = wuk.size(2);
    const int64_t q_rank = gamma1.size(0);
    const int64_t rope_dim = current_rope.size(1);
    TORCH_CHECK(h > 0 && kv_rank > 0 && q_rank > 0 && rope_dim > 0,
                "MLAPO output dimensions must be positive");
    const int64_t quantMode = mla_preprocess_detail::ParseQuantMode(quant_mode);
    mla_preprocess_detail::WarnUnsupportedDimensions(kv_rank, rope_dim);
    CheckDirectMlapoTensor(q_out0, "q_out0", {n, h, kv_rank}, dtype, device);
    CheckDirectMlapoTensor(q_out1, "q_out1", {n, h, rope_dim}, dtype, device);
    CheckDirectMlapoTensor(inner_out, "inner_out", {n, q_rank}, dtype, device);
    CheckDirectMlapoTensor(current_k, "current_k", {n, kv_rank}, dtype, device);
    CheckDirectMlapoTensor(current_rope, "current_rope", {n, rope_dim}, dtype, device);
    return quantMode;
}
}  // namespace vllm_ascend
