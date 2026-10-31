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

#include "register/op_def_registry.h"

namespace ops {
class MlaPreprocess : public OpDef {
public:
    explicit MlaPreprocess(const char *name) : OpDef(name)
    {
        this->Input("hiddenState")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantScale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantOffset1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wdqkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("bias1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantScale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantOffset2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("keycache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("slotMapping")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wuq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("bias2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wuk")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // descale1/descale2 用来把两次 INT8 矩阵乘法的结果还原为浮点数。
        // BF16 路径传入普通 FP32 scale；已有 FP16 路径用 INT64 张量承载约定的反量化位模式。
        // 所以这两个输入都增加 DT_INT64。这里改变的是可接收的参数格式，不是把 scale 数值取整，
        // 也不决定是否输出 innerOut 或是否开启 Direct。
        this->Input("descale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ctkvScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qnopeScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("keycacheOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("q2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("keycacheOut2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("innerOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("cacheMode").AttrType(OPTIONAL).Int(1);
        this->Attr("quantMode").AttrType(OPTIONAL).Int(1);
        // enableInnerOut=true：额外写出 Q 归一化后的中间结果到 innerOut；false：不写这份中间输出。
        // 关闭 enableInnerOut 仍然计算最终 Q 和 KV，也不改变 KV 的分页/连续存储方式。
        // innerOut 参数在签名中始终存在，但关闭导出时不能把其中的旧内容当成这次计算结果。
        this->Attr("enableInnerOut").AttrType(OPTIONAL).Bool(false);
        this->Attr("enableRope").AttrType(OPTIONAL).Bool(true);
        this->Attr("kvCacheStride0").AttrType(REQUIRED).Int(0);
        this->Attr("kvCacheRopeStride0").AttrType(REQUIRED).Int(0);
        // directOutput 控制 KV 的 NPU 输出位置，和上面的 enableInnerOut 分别控制不同的数据：
        // false：KV 按 slot 写分页缓存；true：输入第 i 个 token 的 KV 写到连续输出的第 i 行。
        // CPU 写回还需要 writebackKBase/writebackRopeBase（映射 CPU 池的起始地址）和 writebackCapacity。
        // 三项全为 0：不更新 CPU 池，但仍按 directOutput 生成 NPU 结果；开启时三项须完整有效。
        // 例如启用 Direct 且 slot=17，NPU 输出第 0 行可同时写到 CPU 第 17 行；位置来自 slot 输入。
        this->Attr("directOutput").AttrType(OPTIONAL).Bool(false);
        this->Attr("writebackKBase").AttrType(OPTIONAL).Int(0);
        this->Attr("writebackRopeBase").AttrType(OPTIONAL).Int(0);
        this->Attr("writebackCapacity").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "mla_preprocess");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(MlaPreprocess);
}  // namespace ops
