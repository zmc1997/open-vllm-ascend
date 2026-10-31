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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "mla_preprocess_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {

constexpr size_t INPUT_HIDDEN_STATE = 0;
constexpr size_t INPUT_WDQKV = 3;
constexpr size_t INPUT_GAMMA1 = 5;
constexpr size_t INPUT_KV_CACHE = 14;
constexpr size_t INPUT_SLOT_MAPPING = 15;
constexpr size_t INPUT_WUK = 18;
constexpr size_t OUTPUT_KV_CACHE_ROPE = 3;

constexpr size_t ATTR_CACHE_MODE = 0;
constexpr size_t ATTR_QUANT_MODE = 1;
constexpr size_t ATTR_ENABLE_INNER_OUT = 2;
constexpr size_t ATTR_ENABLE_ROPE = 3;
constexpr size_t ATTR_KV_CACHE_STRIDE0 = 4;
constexpr size_t ATTR_KV_CACHE_ROPE_STRIDE0 = 5;
// Host 通过属性在列表中的位置取值。新增四项接在原有六项之后，必须与 OpDef 和 ACLNN 调用对齐。
// 例如 ATTR_WRITEBACK_CAPACITY=9 对应的是 CPU 池行容量，不能误读成前面的输出步长或地址。
constexpr size_t ATTR_DIRECT_OUTPUT = 6;
constexpr size_t ATTR_WRITEBACK_K_BASE = 7;
constexpr size_t ATTR_WRITEBACK_ROPE_BASE = 8;
constexpr size_t ATTR_WRITEBACK_CAPACITY = 9;

constexpr int64_t CACHE_MODE_MIN = 0;
constexpr int64_t CACHE_MODE_MAX = 3;
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 3;

struct MlaPreprocessCompileInfo {};

bool IsPositiveU32(int64_t value)
{
    return value > 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool IsPhysicalNzCache(const gert::Shape &shape, int64_t cacheMode)
{
    const bool isNzCache = cacheMode == 2 || cacheMode == 3;
    return isNzCache && shape.GetDimNum() == 4 && shape.GetDim(2) != 1;
}

ge::graphStatus ParseShapesAndAttrs(gert::TilingContext *context, mlapo::OpParam &opParam)
{
    // Host 是在 CPU 上运行的算子准备代码，负责把张量形状和选项整理成 Kernel 的执行配置。
    // quantMode 选计算方式；enableInnerOut 选是否额外导出 Q 中间结果；directOutput 选 KV 输出布局。
    // 这三项各管一件事。CPU 写回还需要有效池地址与容量，不能只看某个输出开关就决定。
    // 下面先共用原输入检查，再检查 Direct 的连续输出；实际矩阵计算要等 NPU Kernel 启动后执行。
    const gert::StorageShape *hiddenStorageShape = context->GetInputShape(INPUT_HIDDEN_STATE);
    const gert::StorageShape *wdqkvStorageShape = context->GetInputShape(INPUT_WDQKV);
    const gert::StorageShape *gamma1StorageShape = context->GetInputShape(INPUT_GAMMA1);
    const gert::StorageShape *kvCacheStorageShape = context->GetInputShape(INPUT_KV_CACHE);
    const gert::StorageShape *wukStorageShape = context->GetInputShape(INPUT_WUK);
    const gert::StorageShape *kvCacheRopeStorageShape = context->GetOutputShape(OUTPUT_KV_CACHE_ROPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wdqkvStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma1StorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wukStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheRopeStorageShape);

    const gert::Shape &hiddenShape = hiddenStorageShape->GetOriginShape();
    const gert::Shape &gamma1Shape = gamma1StorageShape->GetOriginShape();
    const gert::Shape &kvCacheShape = kvCacheStorageShape->GetOriginShape();
    const gert::Shape &wukShape = wukStorageShape->GetOriginShape();
    const gert::Shape &kvCacheRopeShape = kvCacheRopeStorageShape->GetOriginShape();

    OP_CHECK_IF(hiddenShape.GetDimNum() != 2,
                OP_LOGE(context, "hiddenState must have rank 2, got %zu.", hiddenShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma1Shape.GetDimNum() < 1,
                OP_LOGE(context, "gamma1 must have at least one dimension."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(wukShape.GetDimNum() != 3,
                OP_LOGE(context, "wuk must have logical rank 3, got %zu.", wukShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kvCacheShape.GetDimNum() < 2,
                OP_LOGE(context, "kv_cache must have at least two dimensions."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kvCacheRopeShape.GetDimNum() < 1,
                OP_LOGE(context, "kv_cache_rope must not be a scalar."), return ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *cacheModePtr = attrs->GetInt(ATTR_CACHE_MODE);
    const int64_t *quantModePtr = attrs->GetInt(ATTR_QUANT_MODE);
    const bool *enableInnerOutPtr = attrs->GetBool(ATTR_ENABLE_INNER_OUT);
    const bool *enableRopePtr = attrs->GetBool(ATTR_ENABLE_ROPE);
    const int64_t *kvCacheStride0Ptr = attrs->GetInt(ATTR_KV_CACHE_STRIDE0);
    const int64_t *kvCacheRopeStride0Ptr = attrs->GetInt(ATTR_KV_CACHE_ROPE_STRIDE0);
    const bool *directOutputPtr = attrs->GetBool(ATTR_DIRECT_OUTPUT);
    const int64_t *writebackKBasePtr = attrs->GetInt(ATTR_WRITEBACK_K_BASE);
    const int64_t *writebackRopeBasePtr = attrs->GetInt(ATTR_WRITEBACK_ROPE_BASE);
    const int64_t *writebackCapacityPtr = attrs->GetInt(ATTR_WRITEBACK_CAPACITY);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, enableInnerOutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, enableRopePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheStride0Ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheRopeStride0Ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, directOutputPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, writebackKBasePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, writebackRopeBasePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, writebackCapacityPtr);

    const int64_t cacheMode = *cacheModePtr;
    const int64_t quantMode = *quantModePtr;
    OP_CHECK_IF(cacheMode < CACHE_MODE_MIN || cacheMode > CACHE_MODE_MAX,
                OP_LOGE(context, "cacheMode must be in [0, 3], got %ld.", cacheMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode < QUANT_MODE_MIN || quantMode > QUANT_MODE_MAX,
                OP_LOGE(context, "quantMode must be in [0, 3], got %ld.", quantMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*kvCacheStride0Ptr <= 0 || *kvCacheRopeStride0Ptr <= 0,
                OP_LOGE(context, "cache dim0 strides must be positive."), return ge::GRAPH_FAILED);

    const int64_t tokenNum = hiddenShape.GetDim(0);
    const int64_t hiddenStateDim = hiddenShape.GetDim(1);
    const int64_t headNum = wukShape.GetDim(0);
    const int64_t qkNopeHeadDim = wukShape.GetDim(1);
    const int64_t kvLoraRank = wukShape.GetDim(2);
    const int64_t qLoraRank = gamma1Shape.GetDim(0);
    OP_CHECK_IF(!IsPositiveU32(tokenNum) || !IsPositiveU32(hiddenStateDim) || !IsPositiveU32(headNum) ||
                    !IsPositiveU32(qkNopeHeadDim) || !IsPositiveU32(kvLoraRank) || !IsPositiveU32(qLoraRank),
                OP_LOGE(context, "MLA model and token dimensions must be positive uint32 values."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenNum > static_cast<int64_t>(mlapo::MAX_SUPPORT_TOKEN_NUMS),
                OP_LOGE(context, "token count must be in [1, %u], got %ld.",
                        mlapo::MAX_SUPPORT_TOKEN_NUMS, tokenNum),
                return ge::GRAPH_FAILED);

    // 旧缓存按“页、页内位置、维度”存储，必须知道一页有多少个 token。
    // Direct 输出只有 [本批 token 数, 每行维度]，第二维是特征宽度，不能再拿它当 block size。
    // 因而 Direct 在这里将分页大小设为 0，后续实际写出按 token 行号处理。
    const bool directOutput = *directOutputPtr;
    const int64_t kvCacheBlockSize =
        directOutput ? 0 : (IsPhysicalNzCache(kvCacheShape, cacheMode) ? kvCacheShape.GetDim(2)
                                                                   : kvCacheShape.GetDim(1));
    int64_t qkRopeHeadDim = kvCacheRopeShape.GetDim(kvCacheRopeShape.GetDimNum() - 1);
    if (IsPhysicalNzCache(kvCacheRopeShape, cacheMode)) {
        qkRopeHeadDim = kvCacheRopeShape.GetDim(1) * kvCacheRopeShape.GetDim(3);
    }
    OP_CHECK_IF((!directOutput && !IsPositiveU32(kvCacheBlockSize)) || !IsPositiveU32(qkRopeHeadDim),
                OP_LOGE(context, "cache block size and RoPE head dimension must be positive uint32 values."),
                return ge::GRAPH_FAILED);

    const auto *hiddenDesc = context->GetInputDesc(INPUT_HIDDEN_STATE);
    const auto *wdqkvDesc = context->GetInputDesc(INPUT_WDQKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, wdqkvDesc);
    const ge::DataType hiddenDtype = hiddenDesc->GetDataType();
    OP_CHECK_IF(hiddenDtype != ge::DT_FLOAT16 && hiddenDtype != ge::DT_BF16,
                OP_LOGE(context, "hiddenState must be float16 or bfloat16."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode == static_cast<int64_t>(mlapo::QuantMode::PER_TOKEN_SYMM_QUANT) &&
                    hiddenDtype != ge::DT_BF16,
                OP_LOGE(context, "per_token_quant_symm supports only bfloat16 hiddenState."),
                return ge::GRAPH_FAILED);

    opParam.hiddenStateDim = static_cast<uint32_t>(hiddenStateDim);
    opParam.N = static_cast<uint32_t>(tokenNum);
    opParam.headNum = static_cast<uint32_t>(headNum);
    opParam.cacheMode = static_cast<int32_t>(cacheMode);
    opParam.quantMode = static_cast<mlapo::QuantMode>(quantMode);
    opParam.inDtype = hiddenDtype == ge::DT_BF16 ? mlapo::OpParam::InputDtype::BFLOAT16
                                                 : mlapo::OpParam::InputDtype::FLOAT16;
    opParam.enableInnerOut = *enableInnerOutPtr;
    opParam.enableRope = *enableRopePtr;
    opParam.directOutput = directOutput;
    // CPU 写回必须成套配置：两个池地址都有效、capacity>0，并且 Direct 输出已开启。
    // 全零表示不写回，例如其他 TP rank 只生成自己的 NPU 输出；只给一个地址属于配置错误。
    // 地址还要满足 32 字节对齐，因为设备端按底层搬运指令的要求访问这块映射内存。
    const int64_t kBase = *writebackKBasePtr;
    const int64_t ropeBase = *writebackRopeBasePtr;
    const int64_t capacity = *writebackCapacityPtr;
    OP_CHECK_IF(kBase < 0 || ropeBase < 0 || capacity < 0,
                OP_LOGE(context, "Direct CPU writeback bases and capacity must be non-negative."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((capacity == 0 && (kBase != 0 || ropeBase != 0)) ||
                    (capacity != 0 && (!directOutput || kBase == 0 || ropeBase == 0 ||
                                      kBase % 32 != 0 || ropeBase % 32 != 0)),
                OP_LOGE(context, "CPU writeback requires Direct mode and aligned mapped pool bases."),
                return ge::GRAPH_FAILED);
    if (directOutput) {
        // Direct changes output destinations, not the model's compute shape.
        // Derive output strides and CPU byte ranges from the actual dimensions.
        // 当前 Direct 接口固定交付“最终 Q + Q 中间结果 + 当前 KV”，所以要求 enableInnerOut=true。
        // 这是 Direct 接口的参数约定，不是说导出 Q 中间结果只能依赖 Direct；普通分页入口也有 inner 模式。
        // Direct 的 current_k/current_rope 必须各有 tokenNum 行，K 每行宽度等于 kvLoraRank。
        // 下面还检查维度相加、头数乘维度等计算，防止设备端 uint32 索引溢出。
        const int64_t dimensionLimit = std::numeric_limits<uint32_t>::max();
        const bool supportedWeights =
            (quantMode == 0 || quantMode == 1) && wdqkvDesc->GetDataType() == ge::DT_INT8;
        OP_CHECK_IF(!supportedWeights || cacheMode != 1 || !*enableInnerOutPtr ||
                        kvCacheShape.GetDimNum() != 2 || kvCacheShape.GetDim(0) != tokenNum ||
                        kvCacheShape.GetDim(1) != kvLoraRank || kvCacheRopeShape.GetDimNum() != 2 ||
                        kvCacheRopeShape.GetDim(0) != tokenNum ||
                        qLoraRank > dimensionLimit - kvLoraRank - qkRopeHeadDim ||
                        qkNopeHeadDim > dimensionLimit - qkRopeHeadDim ||
                        headNum > dimensionLimit / (qkNopeHeadDim + qkRopeHeadDim) ||
                        headNum > dimensionLimit / kvLoraRank || headNum > dimensionLimit / tokenNum,
                    OP_LOGE(context, "Direct MLA requires supported weights, separate contiguous KV and inner output."),
                    return ge::GRAPH_FAILED);
        // 开启 CPU 写回时，一行输入必须对应一个 slot，例如 slots=[17,42] 表示更新 CPU 池的 17/42 行。
        // Host 这里只检查 slots 的形状，因为它不会把设备张量读回 CPU；正负和越界由 Kernel 检查。
        // slots 内容可在图重放前原地更新，所以同一张图能够服务不同请求和不同 CPU 池位置。
        if (capacity != 0) {
            const gert::StorageShape *slotStorage = context->GetInputShape(INPUT_SLOT_MAPPING);
            OP_CHECK_NULL_WITH_CONTEXT(context, slotStorage);
            const gert::Shape &slotShape = slotStorage->GetOriginShape();
            OP_CHECK_IF(slotShape.GetDimNum() != 1 || slotShape.GetDim(0) != tokenNum,
                        OP_LOGE(context, "Direct CPU writeback requires one dynamic slot per token."),
                        return ge::GRAPH_FAILED);
        }
        // The shared dtype check above permits only float16 and bfloat16.
        // FP16/BF16 每个元素都是两字节，池范围按“容量 * 每行维度 * 两字节”计算。
        // 先检查地址乘加不溢出，再检查 K 与 RoPE 的两个地址范围不重叠，避免相互覆盖结果。
        constexpr int64_t elementBytes = sizeof(uint16_t);
        const int64_t kBytes = kvLoraRank * elementBytes;
        const int64_t ropeBytes = qkRopeHeadDim * elementBytes;
        const int64_t limit = std::numeric_limits<int64_t>::max();
        OP_CHECK_IF(capacity > (limit - kBase) / kBytes || capacity > (limit - ropeBase) / ropeBytes,
                    OP_LOGE(context, "Direct CPU writeback address range overflows int64."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(capacity != 0 && !(kBase + capacity * kBytes <= ropeBase ||
                                      ropeBase + capacity * ropeBytes <= kBase),
                    OP_LOGE(context, "Direct CPU K and RoPE pool ranges overlap."),
                    return ge::GRAPH_FAILED);
    }
    opParam.writebackKBase = static_cast<uint64_t>(kBase);
    opParam.writebackRopeBase = static_cast<uint64_t>(ropeBase);
    opParam.writebackCapacity = static_cast<uint64_t>(capacity);
    opParam.qLoraRank = static_cast<uint32_t>(qLoraRank);
    opParam.qkNopeHeadDim = static_cast<uint32_t>(qkNopeHeadDim);
    opParam.qkRopeHeadDim = static_cast<uint32_t>(qkRopeHeadDim);
    opParam.kvLoraRank = static_cast<uint32_t>(kvLoraRank);
    opParam.kvCacheBlockSize = static_cast<uint64_t>(kvCacheBlockSize);
    opParam.kvCacheStride0 = static_cast<uint64_t>(*kvCacheStride0Ptr);
    opParam.kvCacheRopeStride0 = static_cast<uint64_t>(*kvCacheRopeStride0Ptr);
    const ge::DataType weightDtype = wdqkvDesc->GetDataType();
    opParam.isWeightQuantized = weightDtype == ge::DT_FLOAT16 || weightDtype == ge::DT_BF16 ? 0U : 1U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillPlatformInfo(gert::TilingContext *context, mlapo::PlatformInfo &platformInfo,
                                 uint64_t &systemWorkspaceSize)
{
    fe::PlatFormInfos *platform = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform);
    const platform_ascendc::PlatformAscendC ascendcPlatform(platform);
    platformInfo.coreNum = ascendcPlatform.GetCoreNum();
    platformInfo.coreNumAic = ascendcPlatform.GetCoreNumAic();
    platformInfo.coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);
    systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(platformInfo.coreNumAic == 0 || platformInfo.coreNumAiv == 0 || platformInfo.ubSize == 0 ||
                    platformInfo.l0cSize == 0,
                OP_LOGE(context, "invalid platform information for MLA preprocessing."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMlaPreprocess(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlaPreprocess", "TilingContext is null."),
                return ge::GRAPH_FAILED);

    mlapo::OpParam opParam{};
    OP_CHECK_IF(ParseShapesAndAttrs(context, opParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to parse MLA shapes, dtypes or attrs."),
                return ge::GRAPH_FAILED);

    mlapo::PlatformInfo platformInfo{};
    uint64_t systemWorkspaceSize = 0;
    OP_CHECK_IF(FillPlatformInfo(context, platformInfo, systemWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to query platform information."), return ge::GRAPH_FAILED);

    MlaTilingData *tilingData = context->GetTilingData<MlaTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = MlaTilingData{};
    mlapo::MlaPreprocessTiling mlaTiling(platformInfo, opParam, tilingData);
    mlaTiling.Init();

    OP_CHECK_IF(tilingData->userWorkspaceSize > std::numeric_limits<uint64_t>::max() - systemWorkspaceSize,
                OP_LOGE(context, "workspace size overflow."), return ge::GRAPH_FAILED);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(systemWorkspaceSize + tilingData->userWorkspaceSize);
    context->SetBlockDim(platformInfo.coreNumAic);
    // Compile and select one entry for each existing MLA mode key.
    // key 是一个模式编号，告诉 CANN 本次要加载哪一份已编译 Kernel，不能把它当成新计算步骤。
    // 例如 1793 同时记录 BF16 静态量化、K/RoPE 分开存、导出 Q 中间结果和 Direct 连续 KV 输出。
    // CANN 为 TILING_KEY_IS 中的模式分别编译，Host 在启动前选一个版本；本次只运行被选中的版本。
    // 设备 tiling 的 key 与 SetTilingKey 必须一致，否则可能“加载一种程序，却传另一种配置”。
    context->SetTilingKey(tilingData->tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseMlaPreprocess(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(MlaPreprocess)
    .Tiling(TilingMlaPreprocess)
    .TilingParse<MlaPreprocessCompileInfo>(TilingParseMlaPreprocess);

}  // namespace optiling
