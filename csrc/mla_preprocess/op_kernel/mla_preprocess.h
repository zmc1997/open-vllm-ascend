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

#ifndef __MLA_PREPROCESS_H__
#define __MLA_PREPROCESS_H__

// sync
constexpr int32_t QUANT1 = 1;
constexpr int32_t MM1 = 2;
constexpr int32_t MM1QUANT = 3;
constexpr int32_t RMSNORMQUANT2 = 4;
constexpr int32_t MM2 = 5;
constexpr int32_t MM2QUANT = 6;
constexpr int32_t BMM3 = 7;
constexpr int32_t BMM3SPLIT = 8;
constexpr int32_t MM2OUT = 9;
constexpr int32_t EINSUMOUT = 11;
constexpr int32_t EINSUMQUANT = 12;

// ropeConcat
constexpr uint32_t ELE_NUM_FP16 = 16;         // nums of fp16 elements in one block
constexpr uint32_t ELE_NUM_FP32 = 8;          // nums of fp32 elements in one block
constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8;  // stride, 8 * 32 = 256

// rmsNormQuant
constexpr int32_t NUM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr float ZERO = 0;
constexpr uint32_t BUF_FACTOR = 3;         // 1(g) + 1(sqx) + 1(sum) = 3
constexpr uint32_t OFFSET_GAMMA = 0;       // the offset of gamma is 0
constexpr uint32_t OFFSET_SQX = 1;         // the offset of sqx is 1
constexpr uint32_t OFFSET_SUM = 2;         // the offset of sum is 2
constexpr uint32_t OFFSET_WORKSPACE = 3;   // the offset of workspace is 3
constexpr uint32_t REPEAT_TIME_256 = 256;  // 128 default stride
constexpr uint32_t REPEAT_TIME_128 = 128;  // 128 default stride
constexpr uint32_t REPEAT_TIME_64 = 64;    // 64 default stride

constexpr uint8_t CACHE_MODE_KVCACHE = 0;       // single input single output
constexpr uint8_t CACHE_MODE_KROPE_CTKV = 1;    // double in and double out
constexpr uint8_t CACHE_MODE_INT8_NZCACHE = 2;  // high performance KV NZ format/quant int8
constexpr uint8_t CACHE_MODE_NZCACHE = 3;

// pp matmul
constexpr uint32_t FLOAT_BLOCK_SIZE = 64;
constexpr uint32_t HALF_BLOCK_SIZE = 64;
constexpr uint32_t HALF_VECTOR_SIZE = 64;
constexpr uint32_t MM1_OUT_SIZE = 2112;
constexpr uint32_t SPLIT_SIZE_ONE = 576;
constexpr uint32_t SPLIT_SIZE_TWO = 1536;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_ONE = 512;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_TWO = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_ONE = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_TWO = 128;

constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;   // 32 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;  // 256 KB
constexpr uint64_t BLOCK_SIZE_16 = 16;
constexpr uint64_t BLOCK_SIZE_32 = 32;
constexpr uint64_t CUBE_MATRIX_SIZE_512 = 16 * 32;  // 16 * 23
constexpr uint64_t FB_BUFF_SIZE = 1024 * 7;
constexpr uint64_t SCALE_L1_LEN = 4096;
constexpr uint64_t BIAS_L1_LEN = 2048;

constexpr uint64_t CONST_0 = 0;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_8 = 8;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;

// ropeConcat
constexpr uint32_t ROPE_CONCAT_NUM_BUFFER = 2;

// rmsNormQuant
constexpr uint32_t OFFSET_ABS = 3;             // the offset of abs is 3
constexpr uint32_t OFFSET_WORKSPACE_BF16 = 4;  // the offset of workspace is 4

// sync bf16
constexpr int32_t AIC_MM1_START = 2;
constexpr int32_t AIC_MM3_START = 3;
constexpr int32_t AIC_MM2_START = 6;
constexpr int32_t MMAIC = 7;
constexpr int32_t MMAIV = 8;

constexpr uint32_t MAX_HW_SYNC_COUNTER = 5;
constexpr uint32_t SYNC_MODE = 2;

// The tested CANN 9.1.0 precompiler accepts flat numeric macros in TILING_KEY_IS.
// These values preserve the existing dtype, cache, quantization and output bits.
// tilingKey 是 Host 和 Kernel 约定的“模式编号”，数值的不同位分别记录一项配置：
// cacheMode 在低位，quantMode 从 bit 3 开始，bit 8 表示 BF16，bit 9 表示额外导出 innerOut/q_c，
// bit 10 表示 Direct KV 行输出。innerOut 是 Q 归一化后、量化前的结果；bit 9 为 0 不代表不计算 Q。
// Host 的 enableInnerOut 属性编码到 bit 9，directOutput 属性编码到 bit 10，再由完整 key 选择模板实例。
// 例如 1793 = 1024 + 512 + 256 + 1，表示 Direct + 额外 Q 中间输出 + BF16 + 独立 K/RoPE 布局。
// 静态量化 quantMode=0 不额外加数值；是否写 CPU 不编码在此 key 中，而由 CPU 池地址和容量决定。
// 改成宏不是因为宏比 constexpr 运算快，而是当前 CANN 9.1.0 的预编译处理曾无法识别
// TILING_KEY_IS 中的 constexpr 名称，带括号的组合表达式也失败；展开为数值字面量后构建通过。
// 因而这里保留扁平数值宏，让工具在编译计算代码前就能识别模式；此限制不代表所有 CANN 版本。
// 若以后恢复 constexpr 或组合表达式，需要连同预编译和产物入口选择一起验证，而非只看 C++ 语法。
#define KEY_FP16_CACHEMODE_0_QUANTMODE_0 0U
#define KEY_FP16_CACHEMODE_1_QUANTMODE_0 1U
#define KEY_FP16_CACHEMODE_0_QUANTMODE_0_INNER 512U
#define KEY_FP16_CACHEMODE_1_QUANTMODE_0_INNER 513U
#define KEY_DIRECT_OUTPUT 1024U
#define KEY_BF16_CACHEMODE_0_QUANTMODE_0 256U
#define KEY_BF16_CACHEMODE_1_QUANTMODE_0 257U
#define KEY_BF16_CACHEMODE_3_QUANTMODE_0 259U
#define KEY_BF16_CACHEMODE_0_QUANTMODE_1 264U  // bf16 | cm0 | (PER_TOKEN_SYMM<<3)
#define KEY_BF16_CACHEMODE_1_QUANTMODE_1 265U  // bf16 | cm1 | (PER_TOKEN_SYMM<<3)
#define KEY_BF16_CACHEMODE_2_QUANTMODE_1 266U  // bf16 | cm2 | (PER_TOKEN_SYMM<<3)
#define KEY_BF16_CACHEMODE_3_QUANTMODE_1 267U  // bf16 | cm3 | (PER_TOKEN_SYMM<<3)
#define KEY_BF16_CACHEMODE_1_QUANTMODE_3 281U
#define KEY_BF16_CACHEMODE_0_QUANTMODE_0_INNER 768U
#define KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER 769U
#define KEY_BF16_CACHEMODE_3_QUANTMODE_0_INNER 771U
#define KEY_BF16_CACHEMODE_0_QUANTMODE_1_INNER 776U
#define KEY_BF16_CACHEMODE_1_QUANTMODE_1_INNER 777U
#define KEY_BF16_CACHEMODE_2_QUANTMODE_1_INNER 778U
#define KEY_BF16_CACHEMODE_3_QUANTMODE_1_INNER 779U

// 当前已接入的三个量化 Direct key 都包含 bit 10 和 bit 9：输出本批连续 KV 行，并额外导出 q_c。
// 例如 FP16 静态是 1537，BF16 静态是 1793，BF16 动态再加 (1 << 3)，得到 1801。
// 启动计算时应匹配完整 key；只判断“有 Direct 位”无法知道数据类型和量化实现。
// 这些编号也不表示任意 Direct/innerOut 组合都已开放，实际可用模式以入口中的 TILING_KEY_IS 分支为准。
#define KEY_FP16_CACHEMODE_1_QUANTMODE_0_INNER_DIRECT 1537U
#define KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER_DIRECT 1793U
#define KEY_BF16_CACHEMODE_1_QUANTMODE_1_INNER_DIRECT 1801U

enum class QuantMode : int32_t {
    PER_TENSOR_ASYMM_QUANT = 0,
    PER_TOKEN_SYMM_QUANT,
    PER_TOKEN_ASYMM_QUANT,
    NO_QUANT,
};

#endif  // __MLA_PREPROCESS_H__
