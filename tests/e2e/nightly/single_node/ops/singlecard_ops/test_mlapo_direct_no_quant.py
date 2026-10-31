# 这组测试补上“不做 W8A8 量化”的 BF16 分支，避免只验证量化模式就遗漏另一路计算。
# 原算子 mla_preprocess 的 no_quant 分支写分页 KV；Direct 入口改为连续 token KV，Q 算术不变。
# enable_inner_out=True 只额外保存压缩 Q 到 q_c；False 时仍计算两部分 Q，并生成 KV。
# writeback_enabled=True 再按 slots 更新 CPU 池；False 只禁止额外 CPU 写入，连续 KV 仍要输出。
# token=1/6/128 分别覆盖小批次、多目的槽位和核内循环；enable_rope 决定是否传入 cos/sin。
# CPU 写回测试使用真正可由 NPU 访问的共享池，并在另一条设备执行队列读回核对。
# 图重放时更新输入和槽位，但不更换张量地址，用来发现旧输入或旧目的地址被固化的问题。
# 五项位一致是相对原 MLAPO 的保证，不能当成与 Python Native 路径或整模型精度等价。

import pytest
import torch
import torch_npu

from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.utils import enable_custom_op

from . import test_mlapo_direct_modes as qdown

# fixture 是 pytest 负责准备和回收的测试资源。这里复用量化测试的共享池初始化流程，
# 保证无量化分支同样验证真实设备写入和读回，而不是把普通 CPU 张量假装成可映射内存。
mapped_offload = qdown.mapped_offload

enable_custom_op()


@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("token_num", [1, 6, 128])
@torch.inference_mode()
def test_mla_preprocess_kernel(enable_rope, token_num, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_no_quant_case(enable_rope, token_num)


@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("writeback_enabled", [True, False], ids=["writeback", "peer"])
@torch.inference_mode()
def test_mla_no_quant_mapped_writeback_graph(enable_rope, writeback_enabled, mapped_offload, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_no_quant_case(enable_rope, 6, mapped_offload, writeback_enabled)


def _exercise_no_quant_case(enable_rope, token_num, offload=None, writeback_enabled=True):
    # 先执行 original(inner=False)，保存 q_nope/q_rope 和两份分页 KV，再执行 inner=True。
    # inner=True 多保存归一化后的压缩 Q 到 q_c；两部分 Q 和两份 KV 应与 False 时逐位相同。
    # 然后拿这五份结果对照 Direct，区分“另存 Q 中间结果的影响”和“更改 KV 目的地址的影响”。
    torch.manual_seed(0)
    # Retain the original no_quant test's dimensions. N=6 exercises several
    # destinations; N=128 also exercises per-core loops and local-buffer reuse.
    head_num, hidden_size, q_lora_rank = 2, 7168, 1536
    kv_lora_rank, rope_dim, qk_nope_head_dim = 512, 64, 128
    block_size, dtype = 128, torch.bfloat16
    mm1_out = q_lora_rank + kv_lora_rank + rope_dim
    mm2_out = head_num * (qk_nope_head_dim + rope_dim)
    hidden = torch.randn(token_num, hidden_size, dtype=dtype, device="npu") * 0.1
    wdqkv = torch.randn(1, hidden_size // 16, mm1_out, 16, dtype=dtype, device="npu") * 0.01
    wuq = torch.randn(1, q_lora_rank // 16, mm2_out, 16, dtype=dtype, device="npu") * 0.01
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    cos = torch.rand(token_num, 1, 1, rope_dim, dtype=dtype, device="npu")
    sin = torch.rand_like(cos)
    inputs = (
        hidden,
        wdqkv,
        None,
        torch.ones(q_lora_rank, dtype=dtype, device="npu"),
        None,
        wuq,
        None,
        torch.ones(kv_lora_rank, dtype=dtype, device="npu"),
        cos.view(token_num, rope_dim) if enable_rope else None,
        sin.view(token_num, rope_dim) if enable_rope else None,
        # Original no_quant dispatch uses ND W_UK and ND krope_ctkv caches.
        torch.randn(head_num, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device="npu") * 0.01,
    )
    # The schema accepts None for these tensors but has no default values.
    quant_args = {
        name: None
        for name in (
            "quant_scale0",
            "quant_offset0",
            "bias0",
            "quant_scale1",
            "quant_offset1",
            "bias1",
            "ctkv_scale",
            "q_nope_scale",
        )
    }
    caches = tuple(
        torch.full((1, block_size, width), float("nan"), dtype=dtype, device="npu")
        for width in (kv_lora_rank, rope_dim)
    )
    compact_slots = torch.arange(token_num, dtype=torch.int32, device="npu")
    q_outputs = tuple(
        torch.full(shape, float("nan"), dtype=dtype, device="npu")
        for shape in ((token_num, head_num, kv_lora_rank), (token_num, head_num, rope_dim), (token_num, q_lora_rank))
    )
    reference = (*q_outputs, caches[0].view(-1, kv_lora_rank)[:token_num], caches[1].view(-1, rope_dim)[:token_num])

    def original(inner=True):
        # inner 对应 enable_inner_out/WithInnerOut，控制 q_c 是否另存，不控制 Q 投影是否执行。
        torch.ops._C_ascend.mla_preprocess(
            *inputs,
            *caches,
            compact_slots,
            **quant_args,
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=inner,
            q_out0=q_outputs[0],
            kv_cache_out0=caches[0],
            q_out1=q_outputs[1],
            kv_cache_out1=caches[1],
            inner_out=q_outputs[2],
        )

    original(inner=False)
    torch.npu.synchronize()
    original_outputs = [reference[index].clone() for index in (0, 1, 3, 4)]
    for value in reference:
        value.fill_(float("nan"))
    original()
    torch.npu.synchronize()
    # Exposing q_c must not change the four outputs of the original branch.
    for index, expected in zip((0, 1, 3, 4), original_outputs):
        qdown._assert_same_bits(reference[index], expected, f"no_quant original inner on/off output {index}")

    outputs = tuple(torch.full_like(value, float("nan")) for value in reference)

    direct_inputs = (*inputs[:8], cos if enable_rope else None, sin if enable_rope else None, *inputs[10:])

    def direct(writeback=None, call_inputs=direct_inputs):
        # Direct 入口总是要求 q_c/current_k/current_rope 三份额外可见结果。
        # writeback 未传时 CPU 基址默认为零；传入 writeback 时才按其中的地址、容量和 slots 写池。
        torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            *call_inputs,
            **quant_args,
            quant_mode="no_quant",
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
            **(writeback or {}),
        )

    def check_outputs():
        torch.npu.synchronize()
        for name, actual, expected in zip(("q_nope", "q_rope", "q_c", "current_k", "current_rope"), outputs, reference):
            qdown._assert_same_bits(actual, expected, name)

    # Preserve omitted-slots/2D-RoPE calls while checking metadata slots and
    # original 4D cos/sin with the zero-default CPU writeback destinations.
    direct(call_inputs=inputs)
    check_outputs()
    direct({"writeback_slots": compact_slots})
    check_outputs()
    if offload is None:
        return

    # Guard rows surround the advertised pool capacity so negative and
    # capacity-boundary slots cannot silently corrupt adjacent storage.
    pools = tuple(
        offload.empty([block_size + 2, width], dtype=dtype, pin_memory=True) for width in (kv_lora_rank, rope_dim)
    )
    slots = torch.empty(token_num, dtype=torch.int32, device="npu")
    writeback = {
        "writeback_slots": slots,
        "writeback_k_base": pools[0][1:-1].data_ptr() if writeback_enabled else 0,
        "writeback_rope_base": pools[1][1:-1].data_ptr() if writeback_enabled else 0,
        "writeback_capacity": block_size if writeback_enabled else 0,
    }
    # Use the real CPU-to-NPU offload consumer on a different stream, after
    # the producer's completion event, including both guard rows in the read.
    readback = tuple(torch.empty(value.shape, dtype=dtype, device="npu") for value in pools)
    src = torch.tensor([value.data_ptr() for value in pools], dtype=torch.int64, device="npu")
    dst = torch.tensor([value.data_ptr() for value in readback], dtype=torch.int64, device="npu")
    sizes = torch.tensor([value.numel() * value.element_size() for value in pools], dtype=torch.int32, device="npu")
    count = torch.tensor(2, dtype=torch.int32, device="npu")
    consumer, done = torch.npu.Stream(), torch.npu.Event()
    slot_ids: list[int] = []

    def update(iteration):
        torch.manual_seed(100 + iteration)
        hidden.copy_(torch.randn_like(hidden) * 0.1)
        cos.copy_(torch.rand_like(cos))
        sin.copy_(torch.rand_like(sin))
        slot_ids[:] = [3 + iteration, 7 + iteration, -1, 11 + iteration, 15 + iteration, block_size]
        slots.copy_(torch.tensor(slot_ids, dtype=slots.dtype, device=slots.device))
        for pool in pools:
            pool.fill_(-17)
        for value in (*outputs, *reference):
            value.fill_(float("nan"))
        original()
        torch.npu.synchronize()

    def launch():
        direct(writeback)
        producer = torch.npu.current_stream()
        done.record(producer)
        consumer.wait_event(done)
        with torch.npu.stream(consumer):
            assert offload.sparse_copy(src, dst, sizes, count, hidden.device) in (None, 0)
        producer.wait_stream(consumer)

    def check():
        check_outputs()
        # 对算子公布 capacity=128，实际分配 130 行：中间 128 行可写，首尾各留一行保护。
        # 因此 slot=3 对应整池第 4 行；slot=-1/128 应跳过 CPU 写入，保护行继续保留 -17。
        # writeback_enabled=False 时连合法 slot 都不写；整池比较同时检查禁写和越界保护行为。
        # 另一条设备队列读回的 read 也包含保护行，检查真实消费者看到的值而非只看主机内存。
        for pool, read, value in zip(pools, readback, reference[3:]):
            expected = torch.full_like(pool, -17)
            value = value.cpu()
            for row, slot in enumerate(slot_ids):
                if writeback_enabled and 0 <= slot < block_size:
                    expected[slot + 1] = value[row]
            qdown._assert_same_bits(pool, expected, "CPU pool including guard and untouched rows")
            qdown._assert_same_bits(read, expected, "CPU-to-NPU consumer")

    update(0)
    launch()
    check()
    streams = (torch.npu.Stream(), torch.npu.Stream())
    graphs = []
    for stream in streams:
        graph = UpdatableGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            launch()
        graphs.append(graph)
        check()
    prior_q = outputs[0].cpu().clone()
    for iteration in (1, 2, 3):
        # stream 是设备任务的执行队列；两张图在不同队列之间交替重放，避免只验证单队列情况。
        # 每轮更新输入和 slots 后重新运行原算子作为参考，防止两份捕获图都读旧数据却互相相等。
        update(iteration)
        with torch.npu.stream(streams[iteration % 2]):
            graphs[(iteration + 1) % 2].replay()
        # Fresh eager results prevent two stale captured outputs from passing.
        check()
        assert not torch.equal(outputs[0].cpu().view(torch.int16), prior_q.view(torch.int16))
        prior_q = outputs[0].cpu().clone()
