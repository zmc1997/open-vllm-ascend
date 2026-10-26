# Fine-Grained Tensor Parallelism (Fine-grained TP)

## Overview

Fine-Grained Tensor Parallelism (Fine-grained TP) extends standard tensor parallelism by enabling **independent tensor-parallel sizes for different model components**. Instead of applying a single global `tensor_parallel_size` to all layers, Fine-grained TP allows users to configure separate TP sizes for key modules—such as embedding, language model head (lm_head), attention output projection (o_proj), and MLP blocks—via the `finegrained_tp_config` parameter.

This capability supports heterogeneous parallelism strategies within a single model, providing finer control over weight distribution, memory layout, and communication patterns across devices. The feature is compatible with standard dense transformer architectures and integrates seamlessly into vLLM’s serving pipeline.

---

## Benefits of Fine-grained TP

Fine-Grained Tensor Parallelism delivers two primary performance advantages through targeted weight sharding:

- **Reduced Per-Device Memory Footprint**:  
  Fine-grained TP shards large weight matrices (e.g., LM Head, o_proj) across devices, lowering peak memory usage and enabling larger batches or deployment on memory-limited hardware—without quantization.
  
- **Faster Memory Access in GEMMs**:  
  In decode-heavy workloads, GEMM performance is often memory-bound. Weight sharding reduces per-device weight fetch volume, cutting DRAM traffic and improving bandwidth efficiency—especially for latency-sensitive layers like LM Head and o_proj.

Together, these effects allow practitioners to better balance memory, communication, and compute—particularly in high-concurrency serving scenarios—while maintaining compatibility with standard dense transformer models.

---

## Supported Scenarios

### Models  

Fine-grained TP is **model-agnostic** and supports all standard dense transformer architectures, including Llama, Qwen, DeepSeek (base/dense variants), and others.

### Component & Execution Mode Support  

| TP config     | Eager | Graph | Hybrid | Prefill | Decode |
| ------------- | ----- | ----- | ------ | ------- | ------ |
| **embedding** | ✅     | ✅     | ✅      | ✅       | ✅      |
| **o_proj**    | ❌     | ✅     | ❌      | ❌       | ✅      |
| **mlp**       | ✅     | ✅     | ✅      | ✅       | ✅      |
| **LMhead**    | ✅     | ✅     | ✅      | ✅       | ✅      |

> ⚠️ Note:  
>
> - `o_proj` TP is only supported in Graph mode during Decode, because dummy_run in eager mode will not trigger o_proj.
> - `mlp` TP supports dense models, or dense layers in MoE models. For example, the first three dense layers of DeepSeek-R1.

### Configuration Limit

The Fine-Grained TP size for any component must:

- Be **≤ the data-parallel (DP) size**, and  
- **Evenly divide the DP size** (i.e., `dp_size % tp_size == 0`) to ensure valid device assignment and communication grouping.

> ⚠️ Violating these constraints will result in runtime errors or undefined behavior.

### Experimental GLM o_proj with outer TP

For `glm_moe_dsa` static W8A8/BF16 decode, o_proj can additionally split each
outer TP weight shard across adjacent DP replicas. Here
`oproj_tensor_parallel_size` means the **total** number of weight shards, so
the extra DP group size is `oproj_tensor_parallel_size / tensor_parallel_size`.
The total must be a larger multiple of outer TP, and the extra DP group size
must divide DP. For example, DP16/TP2 with o_proj TP4 uses DP subgroups of two.
LM-head and other fine-grained settings retain their existing meanings.

This experimental path requires a KV consumer, graph mode, recompute scheduler,
PP=PCP=DCP=1, and no DSA-CP or FlashComm1/SP. The main model and eager MTP use
the same o_proj groups; idle DP replicas must join every exchange. Changing
outer TP alone does not enable the feature.

Merge the following into the existing `additional_config`, preserving its
other fields:

```json
{
  "finegrained_tp_config": {
    "lmhead_tensor_parallel_size": 16,
    "oproj_tensor_parallel_size": 4,
    "oproj_communication": "all_reduce"
  }
}
```

| Communication | Operation after DP all-to-all | Status |
| --- | --- | --- |
| `reduce_scatter` (default) | Quantized MM, DP RS, outer TP AR | Reference implementation |
| `all_reduce` | Quantized MM, combined TP×DP AR, select local DP rows | Removes one collective stage |
| `matmul_reduce_scatter` | Quantized MM+DP RS, post-RS compensation, outer TP AR | Experimental, outer TP2 and total TP4 only |

The MMRS candidate uses AIV, `bias=None`, and adds BF16 compensation on every
DP rank with outer TP rank zero. This changes rounding compared with adding
int32 bias inside QuantMM. Numerical and full model accuracy gates are required
before use; CPU routing tests do not validate the device implementation.
None of these paths has an established end-to-end speedup in this patch.

For outer TP2 and total o_proj TP4, `oproj_quant_pack=true` optionally fuses
static W8A8 quantization and the DP send-layout conversion. The default is
`false`. The sender produces contiguous INT8 `[2, N, K/2]`, AllToAll transfers
INT8, and the receiver skips quantization. The kernel uses the stored BF16
reciprocal and offset with FP32 separate multiply/add, round-to-nearest-even
and int8 saturation; quantized matmul compensation and reduction are unchanged.
This option does not select the reduction: explicitly choose `all_reduce`
when comparing the AR4 candidate. Numerical equivalence and complete-chain
performance must be checked on the target image before enabling model tests.

---

## How to Use Fine-grained TP

### Configuration Format

Fine-grained TP is controlled via the `finegrained_tp_config` field inside `--additional-config`.

```bash
--additional-config '{
    "finegrained_tp_config": {
        "embedding_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8
    }
}'
```

### Example Usage

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "finegrained_tp_config": {
            "embedding_tensor_parallel_size": 8,
            "lmhead_tensor_parallel_size": 8,
            "mlp_tensor_parallel_size": 8
        }
    }'
```

---

## Experimental Results

To evaluate the effectiveness of fine-grained TP in large-scale service scenarios, we use the model **DeepSeek-R1-W8A8**, deploy PD separated decode instances in an environment of 32 cards Ascend Atlas A2 inference products*64G (A2), with parallel configuration as DP32+EP32, and fine-grained TP size of 8; the performance data is as follows.

| Module           | Memory Savings | TPOT Impact (batch=24)    |
| ---------------- | -------------- | ------------------------- |
| o_proj TP = 8    | 5.8 GB         | **+1.5 ms** (degradation) |
| LM head TP = 8   | 1.51 GB        | **−1.2 ms** (improvement) |
|  FFN TP = 8 | 0.9 GB         | **−1.0 ms** (improvement) |
| Embedding TP = 8 | 1.51 GB        | **−1.0 ms** (improvement) |
| **Total**        | **9.72 GB**    | —                         |

- We achieved significant gains in terms of high memory capacity on a single card, as well as the benefits of TPOT.

---

## ✅ Deployment Recommendations  

Fine-grained TP is the **most effective** in the **decode instance** of PD separation, where models are typically deployed in all-DP mode. In this setup, sharding weight-heavy layers reduces redundant storage and memory pressure.
