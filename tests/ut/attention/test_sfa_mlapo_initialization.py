"""Post-load scale refresh and observable Direct/Native initialization routing."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401
import vllm.logger as vllm_logger

# Match worker startup: register ops before importing the attention backend.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import sfa_kv_offload as sfa


@pytest.fixture
def initialization(monkeypatch):
    config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=False))
    monkeypatch.setattr(sfa, "get_ascend_config", lambda: config)
    monkeypatch.setenv("VLLM_PATCH_TRACE", "0")
    # The test has no distributed group; exercise the real logger's once cache
    # on the rank that normally emits initialization messages.
    monkeypatch.setattr(vllm_logger, "_should_log_with_scope", lambda scope: True)
    vllm_logger._print_info_once.cache_clear()
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            if "Sparse KV offload decode preprocessing selected:" in record.getMessage():
                records.append(record)

    handler = Capture()
    level = sfa.logger.level
    sfa.logger.setLevel(logging.INFO)
    sfa.logger.addHandler(handler)
    try:
        yield config, records
    finally:
        sfa.logger.removeHandler(handler)
        sfa.logger.setLevel(level)
        vllm_logger._print_info_once.cache_clear()


def make_impl(heads=16):
    # Use the actual class/methods without loading a model or touching an NPU.
    impl = sfa.AscendSFAKVOffloadImpl.__new__(sfa.AscendSFAKVOffloadImpl)
    scheme = sfa.AscendW8A8LinearMethod.__new__(sfa.AscendW8A8LinearMethod)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(6144, 2624)), quant_method=SimpleNamespace(quant_method=scheme)
    )
    impl.q_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))
    impl.enable_mlapo = True
    impl.enable_dsa_cp = impl.enable_sparse_sfa_c8 = False
    impl.num_heads, impl.num_kv_heads = heads, 1
    impl.q_lora_rank, impl.kv_lora_rank = 2048, 512
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = 192, 64
    impl.q_a_layernorm = impl.kv_a_layernorm = object()
    impl.gamma2 = torch.ones(512, dtype=torch.bfloat16)
    impl.quant_scale0 = torch.tensor([0.5], dtype=torch.bfloat16)
    impl.quant_scale1 = torch.tensor([0.25], dtype=torch.bfloat16)
    # Only weight reformatting is replaced; the actual precondition checks,
    # selection, scale refresh, allocation and INFO-once behavior still execute.
    impl._process_weights_for_fused_mlapo = Mock()
    return impl


@pytest.mark.parametrize("heads", [16, 32])
def test_reload_refreshes_fixed_scale_backing_and_logs_once(initialization, heads):
    _, records = initialization
    impl = make_impl(heads)
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    storage = [getattr(impl, f"_mlapo_quant_scale_storage{i}") for i in (0, 1)]
    views = [getattr(impl, f"_mlapo_quant_scale{i}") for i in (0, 1)]
    pointers = [value.data_ptr() for value in storage + views]
    output_pointers = (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())
    impl.quant_scale0 = torch.tensor([0.75], dtype=torch.bfloat16)
    impl.quant_scale1 = torch.tensor([0.125], dtype=torch.bfloat16)
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    assert pointers == [getattr(impl, f"_mlapo_quant_scale_storage{i}").data_ptr() for i in (0, 1)] + [
        getattr(impl, f"_mlapo_quant_scale{i}").data_ptr() for i in (0, 1)
    ]
    assert output_pointers == (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())
    for i, value in enumerate((0.75, 0.125)):
        assert torch.equal(storage[i], torch.full((16,), value, dtype=torch.bfloat16))
        assert views[i].item() == value
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert f"path=Direct heads={heads}" in records[0].getMessage()


@pytest.mark.parametrize(
    "condition, reason",
    [
        ("keep_cache", "keep_device_kv_cache=True"),
        ("disabled", "enable_mlapo=False"),
        ("activation", "activation_dtype=torch.float32"),
        ("down_quant", "fused_qkv_a_proj_quant="),
        ("up_quant", "q_proj_quant="),
        ("num_kv_heads", "num_kv_heads=2"),
        ("kv_lora_rank", "kv_lora_rank=256"),
        ("qk_rope_head_dim", "qk_rope_head_dim=32"),
        ("q_lora_rank", "q_lora_rank=1024"),
        ("qk_nope_head_dim", "qk_nope_head_dim=128"),
        ("hidden_size", "fused_qkv_a_proj_weight_rows=4096"),
        ("dsa_cp", "Fused preprocessing does not support DSA-CP."),
        ("q_norm", "Fused preprocessing requires q_a_layernorm and kv_a_layernorm."),
        ("kv_norm", "Fused preprocessing requires q_a_layernorm and kv_a_layernorm."),
        ("sparse_c8", "MLAPO does not support sparse C8; use PROLOG_V3 instead."),
        ("unsupported", "fused MLAPO unsupported"),
    ],
)
def test_native_fallback_records_real_reason_without_patch_trace(initialization, condition, reason):
    config, records = initialization
    impl = make_impl()
    dtype = torch.bfloat16
    if condition == "keep_cache":
        config.sparse_kv_offload_config.keep_device_kv_cache = True
    elif condition == "disabled":
        impl.enable_mlapo = False
    elif condition == "activation":
        dtype = torch.float32
    elif condition == "down_quant":
        impl.fused_qkv_a_proj.quant_method.quant_method = object()
    elif condition == "up_quant":
        impl.q_proj.quant_method.quant_method = object()
    elif condition in ("num_kv_heads", "kv_lora_rank", "qk_rope_head_dim", "q_lora_rank", "qk_nope_head_dim"):
        setattr(impl, condition, {"num_kv_heads": 2, "kv_lora_rank": 256, "qk_rope_head_dim": 32,
                                 "q_lora_rank": 1024, "qk_nope_head_dim": 128}[condition])
    elif condition == "hidden_size":
        impl.fused_qkv_a_proj.weight.shape = (4096, 2624)
    elif condition == "dsa_cp":
        impl.enable_dsa_cp = True
    elif condition == "q_norm":
        impl.q_a_layernorm = None
    elif condition == "kv_norm":
        impl.kv_a_layernorm = None
    elif condition == "sparse_c8":
        impl.enable_sparse_sfa_c8 = True
    elif condition == "unsupported":
        # Exercise the defensive branch when a future try-enable rejects a
        # configuration without one of today's documented rejection reasons.
        impl._try_enable_type = Mock(return_value=False)
    assert impl._resolve_preprocess_type(dtype) == sfa.PreprocessType.NATIVE
    assert impl._resolve_preprocess_type(dtype) == sfa.PreprocessType.NATIVE
    assert not hasattr(impl, "_mlapo_current_k")
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "path=Native" in records[0].getMessage() and reason in records[0].getMessage()


def test_different_head_configurations_each_emit_initialization_info(initialization):
    _, records = initialization
    for heads in (16, 16, 32, 32):
        assert make_impl(heads)._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    assert len(records) == 2
    assert [f"path=Direct heads={heads}" in record.getMessage() for heads, record in zip((16, 32), records)] == [True, True]
