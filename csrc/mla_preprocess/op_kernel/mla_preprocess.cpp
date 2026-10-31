// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#include "kernel_operator.h"

#include "mla_preprocess_mix_fp16.hpp"
#include "mla_preprocess_mix_bf16.hpp"
#include "mla_preprocess_mix_bf16_qdown.hpp"
#include "mla_preprocess_mix_bf16_nq.hpp"

#include "mla_preprocess_tiling_data.h"

// MLAPO 的设备入口。hiddenState、权重和量化参数由调用方传入，输出缓冲区也由调用方提前分配。
// q、q2 接收最终 Q 的两个分量；启用额外输出时，innerOut（Python 侧的 q_c）接收 Q 归一化后、量化前的结果。
// innerOut 是 Q 的中间结果，不是 K 或 V；不开这个额外输出也要继续计算最终 Q。
// keycacheOut、keycacheOut2 接收 K 和 RoPE：普通模式写 NPU 分页 cache，Direct 模式写本批连续行。
// CPU 写回是另一个选择：调用方还须提供合法的 CPU 池地址、非零行容量和每个 token 的目标 slot。
// 因此选了 Direct 不等于必然写 CPU；这里只消费地址和 slot，不申请池或替业务分配槽位。
extern "C" __global__ __aicore__ void mla_preprocess(
    GM_ADDR hiddenState, GM_ADDR quantScale1, GM_ADDR quantOffset1, GM_ADDR wdqkv,
    GM_ADDR bias1, GM_ADDR gamma2, GM_ADDR beta2, GM_ADDR quantScale2, GM_ADDR quantOffset2, GM_ADDR gamma3,
    GM_ADDR sin1, GM_ADDR cos1, GM_ADDR sin2, GM_ADDR cos2, GM_ADDR keycache, GM_ADDR slotMapping, GM_ADDR wuq,
    GM_ADDR bias2, GM_ADDR wuk, GM_ADDR descale1, GM_ADDR descale2, GM_ADDR ctkvScale, GM_ADDR qnopeScale, GM_ADDR q,
    GM_ADDR keycacheOut, GM_ADDR q2, GM_ADDR keycacheOut2, GM_ADDR innerOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlaTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(2);
#endif

    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
#endif

    GET_TILING_DATA_WITH_STRUCT(MlaTilingData, mlaTilingData, tiling);

    // ACLNN initializes the system-workspace state before entering the kernel.
    // Do not overwrite it with the workspace argument; this operator's custom
    // PpMatmul/FFTS path only uses the user portion.
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GM_ADDR s1 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s1Offset);
    GM_ADDR s2 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s2Offset);
    GM_ADDR s3 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s3Offset);
    GM_ADDR s4 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s4Offset);
    GM_ADDR s5 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s5Offset);

    // tilingKey 是本次计算模式的编号，包含数据类型、量化方式、缓存布局、是否导出 innerOut 和是否 Direct。
    // Host 用 SetTilingKey 选择编号，构建工具用 TILING_KEY_IS 为不同编号生成各自的编译入口。
    // 例如编号 1793 选择 BF16 静态 Direct；这次只运行这个实现，不在一个大 switch 中保留全部计算分支。
    // ProcessCube 交给 AIC 矩阵核做矩阵乘，ProcessVector 交给 AIV 向量核做量化、归一化、RoPE 和搬运。
    // 每组混合核仍是 1 个 AIC 配 2 个 AIV，矩阵结果和向量结果通过原有事件依次交接，属于同一次融合调用。
    if (TILING_KEY_IS(KEY_FP16_CACHEMODE_1_QUANTMODE_0_INNER_DIRECT)) {
        // 此 FP16 实例末尾的参数依次是 DirectOutput=true、WithInnerOut=true。
        // DirectOutput=true：K/RoPE 写本批的连续输出行；CPU 是否同时写回还取决于池地址和容量。
        // WithInnerOut=true：将 Q 归一化后、INT8 量化前的 FP16 值额外保存到 innerOut/q_c，供调用方复用。
        // WithInnerOut 不控制最终 Q 的计算，也不控制 K/RoPE 的输出位置；两种作用不要混在一起理解。
        MLAPO_FP16::MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, true, true>
            opFp16Direct(mlaTilingData, tiling);
        opFp16Direct.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, innerOut);
        if ASCEND_IS_AIC {
            opFp16Direct.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opFp16Direct.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER_DIRECT)) {
        // 编号 1793 已限定 BF16、静态量化、额外 Q 中间输出和 Direct KV 行输出。
        // 内层再检查 KEY_DIRECT_OUTPUT，是为了保留已有性能验证所使用的设备入口结构。
        // 等价源码可能得到不同设备程序；具体编译原因仍未确认，所以删改这个判断后要重新同条件测量。
        // 外层先匹配完整编号，保证 FP16 和动态量化不会仅因带有 Direct 位而误入这个 BF16 静态实现。
        // 此处的 Direct 位只说明 K/RoPE 的 NPU 输出布局；CPU 写回仍须另行提供合法池地址和容量。
        if ((mlaTilingData.tilingKey & KEY_DIRECT_OUTPUT) != 0) {
            MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT, true>
                opBf16DirectQm0(mlaTilingData, tiling);
            opBf16DirectQm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                              quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                              bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                              s1, s2, s3, s4, s5, innerOut);
            if ASCEND_IS_AIC {
                opBf16DirectQm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16DirectQm0.ProcessVector();
            }
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_1_INNER_DIRECT)) {
        // quantMode=PER_TOKEN_SYMM_QUANT 表示按 token 动态量化激活，必须使用对应 BF16 计算实例。
        // DirectOutput=true 让本实例的 K/RoPE 按 token 行号输出；本文件的带 innerOut 实现仍额外导出 q_c。
        // 若合法 CPU 池地址和 writebackCapacity 同时就绪，再按 writeback_slots 额外保存 K/RoPE，
        // 不会把 q_c 当作 KV 写进池。
        MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT, true>
            opBf16DirectQm1(mlaTilingData, tiling);
        opBf16DirectQm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16DirectQm1.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16DirectQm1.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_FP16_CACHEMODE_0_QUANTMODE_0)) {
        // 此实例未传布尔参数，使用 DirectOutput=false、WithInnerOut=false 的默认值：
        // K/RoPE 按 slot 更新 NPU 分页 cache，不额外导出 Q 归一化中间结果，最终 Q 仍照常计算。
        // 后面的其他分页分支也保留各自的计算模板，只把原 switch 选择改为按 key 编译。
        MLAPO_FP16::MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> opFp16Cm0Qm0(
            mlaTilingData, tiling);
        opFp16Cm0Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3);
        if ASCEND_IS_AIC {
            opFp16Cm0Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opFp16Cm0Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_FP16_CACHEMODE_0_QUANTMODE_0_INNER)) {
        // 这是已接入的普通分页 + innerOut 分支，参数为 DirectOutput=false、WithInnerOut=true。
        // K/RoPE 仍按 slot 写 NPU 分页 cache；额外将 Q 归一化后、量化前的 FP16 结果存入 innerOut/q_c。
        // 与上一分支相比，多保存一个 Q 中间结果，最终 Q 的归一化、量化和矩阵计算仍然执行。
        MLAPO_FP16::MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, false, true>
            opFp16Cm0Qm0Inner(mlaTilingData, tiling);
        opFp16Cm0Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, innerOut);
        if ASCEND_IS_AIC {
            opFp16Cm0Qm0Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opFp16Cm0Qm0Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_FP16_CACHEMODE_1_QUANTMODE_0)) {
        MLAPO_FP16::MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND>
            opFp16Cm1Qm0(mlaTilingData, tiling);
        opFp16Cm1Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3);
        if ASCEND_IS_AIC {
            opFp16Cm1Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opFp16Cm1Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_FP16_CACHEMODE_1_QUANTMODE_0_INNER)) {
        MLAPO_FP16::MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, false, true>
            opFp16Cm1Qm0Inner(mlaTilingData, tiling);
        opFp16Cm1Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, innerOut);
        if ASCEND_IS_AIC {
            opFp16Cm1Qm0Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opFp16Cm1Qm0Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_0_QUANTMODE_0)) {
        MLAPO_BF16::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm0Qm0(mlaTilingData, tiling);
        opBf16Cm0Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                        quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                        bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                        s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm0Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm0Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_0)) {
        MLAPO_BF16::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm1Qm0(mlaTilingData, tiling);
        opBf16Cm1Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                        quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                        bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                        s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm1Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm1Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_3_QUANTMODE_0)) {
        MLAPO_BF16::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm3Qm0(mlaTilingData, tiling);
        opBf16Cm3Qm0.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm3Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm3Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_0_QUANTMODE_1)) {
        MLAPO_BF16::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm0Qm1(mlaTilingData, tiling);
        opBf16Cm0Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                        quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                        bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                        s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm0Qm1.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm0Qm1.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_1)) {
        MLAPO_BF16::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm1Qm1(mlaTilingData, tiling);
        opBf16Cm1Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                        quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                        bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                        s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm1Qm1.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm1Qm1.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_2_QUANTMODE_1)) {
        MLAPO_BF16::MLAOperation<__bf16, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm2Qm1(mlaTilingData, tiling);
        opBf16Cm2Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm2Qm1.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm2Qm1.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_3_QUANTMODE_1)) {
        MLAPO_BF16::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm3Qm1(mlaTilingData, tiling);
        opBf16Cm3Qm1.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5);
        if ASCEND_IS_AIC {
            opBf16Cm3Qm1.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm3Qm1.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_3)) {
        MLAPO_BF16_NQ::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND>
            opBf16Cm1Qm0(mlaTilingData, tiling);
        opBf16Cm1Qm0.Init(hiddenState, wdqkv, gamma2, beta2,
                        gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                        wuk, q, keycacheOut, q2, keycacheOut2,
                        s1, s2, s3);
        if ASCEND_IS_AIC {
            opBf16Cm1Qm0.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm1Qm0.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_0_QUANTMODE_0_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm0Qm0Inner(mlaTilingData, tiling);
        opBf16Cm0Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm0Qm0Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm0Qm0Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm1Qm0Inner(mlaTilingData, tiling);
        opBf16Cm1Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm1Qm0Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm1Qm0Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_3_QUANTMODE_0_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TENSOR_ASYMM_QUANT>
            opBf16Cm3Qm0Inner(mlaTilingData, tiling);
        opBf16Cm3Qm0Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm3Qm0Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm3Qm0Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_0_QUANTMODE_1_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm0Qm1Inner(mlaTilingData, tiling);
        opBf16Cm0Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm0Qm1Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm0Qm1Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_1_QUANTMODE_1_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm1Qm1Inner(mlaTilingData, tiling);
        opBf16Cm1Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm1Qm1Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm1Qm1Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_2_QUANTMODE_1_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm2Qm1Inner(mlaTilingData, tiling);
        opBf16Cm2Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm2Qm1Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm2Qm1Inner.ProcessVector();
        }
    } else if (TILING_KEY_IS(KEY_BF16_CACHEMODE_3_QUANTMODE_1_INNER)) {
        MLAPO_BF16_INNER::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                 QuantMode::PER_TOKEN_SYMM_QUANT>
            opBf16Cm3Qm1Inner(mlaTilingData, tiling);
        opBf16Cm3Qm1Inner.Init(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2,
                          quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq,
                          bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2,
                          s1, s2, s3, s4, s5, innerOut);
        if ASCEND_IS_AIC {
            opBf16Cm3Qm1Inner.ProcessCube();
        }
        if ASCEND_IS_AIV {
            opBf16Cm3Qm1Inner.ProcessVector();
        }
    }
}
