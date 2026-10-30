# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefetch GLM target decode and eager MTP weights at existing execution windows.

Plans belong to a loaded target model, never to a process-global singleton.
Hooks run inside existing MoE/MLA custom ops, so Dynamo cannot remove
the stream operations. ACL capture records the event chain for every replay.
"""

from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MLAPO_MAX_SUPPORTED_TOKENS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_config import WeightPrefetchConfig


class DirectWeightPrefetchPlan:
    """One producer MoE and its immediately following Direct consumer.

    ``submitted`` pairs Python eager/warmup/capture calls. During replay the
    captured events execute on device; Python flags are not involved. A model
    worker must serialize target forwards (DBO is deliberately unsupported).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        layer_name: str,
        stream: Any,
        max_prefetch_bytes: int,
        wu_q_weight: torch.Tensor | None = None,
        wu_q_max_prefetch_bytes: int = 0,
        w_uk_t_weight: torch.Tensor | None = None,
        w_uk_t_max_prefetch_bytes: int = 0,
    ):
        self.weight = weight
        self.layer_name = layer_name
        self.stream = stream
        # The transformed NZ tensor owns its storage. Use physical bytes,
        # which can differ from numel * element_size for padded layouts.
        self.prefetch_bytes = min(weight.untyped_storage().nbytes(), max_prefetch_bytes)
        self.weights = [(weight, self.prefetch_bytes)] if self.prefetch_bytes > 0 else []
        if wu_q_weight is not None and wu_q_max_prefetch_bytes > 0:
            self.weights.append((wu_q_weight, min(wu_q_weight.untyped_storage().nbytes(), wu_q_max_prefetch_bytes)))
        if w_uk_t_weight is not None and w_uk_t_max_prefetch_bytes > 0:
            self.weights.append(
                (w_uk_t_weight, min(w_uk_t_weight.untyped_storage().nbytes(), w_uk_t_max_prefetch_bytes))
            )
        self.producer_ready = torch_npu.npu.Event()
        self.prefetch_done = torch_npu.npu.Event()
        self.submitted = False

    def _can_submit(self, dependency: torch.Tensor) -> bool:
        if not is_forward_context_available():
            return False
        context = get_forward_context()
        if getattr(context, "is_draft_model", False) or getattr(context, "in_profile_run", False):
            return False
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            return False
        metadata = metadata.get(self.layer_name)
        return (
            metadata is not None
            and metadata.attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            and getattr(metadata, "num_prefills", 0) == 0
            and 0 < dependency.shape[0] <= MLAPO_MAX_SUPPORTED_TOKENS
        )

    def submit(self, dependency: torch.Tensor) -> None:
        self.submitted = False
        if not self._can_submit(dependency):
            return
        # Wait only for the producer stream's work up to this point:
        # shared-expert down for Direct or V-up output for o_proj. A Direct plan
        # must not join the main stream, which would also wait for Combine.
        self.producer_ready.record(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(self.stream):
            self.stream.wait_event(self.producer_ready)
            for weight, prefetch_bytes in self.weights:
                torch_npu.npu_prefetch(weight, dependency, prefetch_bytes)
            # One completion event covers all selected weights before Direct consumes them.
            self.prefetch_done.record(self.stream)
            # Eager execution may otherwise recycle the temporary dependency
            # while the prefetch stream still references it.
            if not getattr(get_forward_context(), "capturing", False):
                dependency.record_stream(self.stream)
        self.submitted = True

    def wait(self) -> None:
        if self.submitted:
            torch_npu.npu.current_stream().wait_event(self.prefetch_done)
            self.submitted = False


class OProjWeightPrefetchPlan(DirectWeightPrefetchPlan):
    """Single-weight V-up -> input quantization -> static W8A8 o_proj plan."""

    def _can_submit(self, dependency: torch.Tensor) -> bool:
        # Fused matmul/reduce-scatter bypasses the static linear scheme's
        # consumer hook. Never submit a prefetch without a matching wait.
        # Already-quantized input has no quantization window to hide prefetch.
        return (
            dependency.dtype != torch.int8
            and super()._can_submit(dependency)
            and not getattr(get_forward_context(), "mmrs_fusion", False)
        )


def bind_direct_weight_prefetch(
    model: torch.nn.Module, vllm_config: "VllmConfig", config: "WeightPrefetchConfig"
) -> list[DirectWeightPrefetchPlan]:
    """Bind final weight storage after loading, before warmup/graph capture.

    The first version supports only target GLM SFA offload on decode workers.
    PP boundaries, dense layers, native preprocessing and unsupported paths
    receive no Direct plan and retain their existing execution. Optional
    same-layer o_proj plans use a separate stream and consumer wait.
    """
    if not config.enabled:
        return []

    # Lazy imports: binding happens only after all weight postprocessing and
    # Ascend model/attention construction have completed.
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
    from vllm_ascend.attention.sfa_v1 import PreprocessType
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.quantization.quant_type import QuantType

    ascend_config = get_ascend_config()
    parallel = vllm_config.parallel_config
    transfer = vllm_config.kv_transfer_config
    if not (
        vllm_config.model_config.hf_text_config.model_type == "glm_moe_dsa"
        and transfer is not None
        and transfer.is_kv_consumer
        and not transfer.is_kv_producer
        and parallel.tensor_parallel_size > 1
        and not parallel.enable_dbo
        and vllm_config.lora_config is None
        and ascend_config.sparse_kv_offload_config.enabled
        and ascend_config.multistream_overlap_shared_expert
    ):
        logger.warning("Direct weight prefetch skipped: unsupported model or execution configuration.")
        return []
    if not callable(getattr(torch_npu, "npu_prefetch", None)):
        raise RuntimeError("weight_prefetch_config requires torch_npu.npu_prefetch")

    plans = []
    streams: dict[torch.device, Any] = {}
    for backbone in model.modules():
        if not isinstance(backbone, DeepseekV2Model):
            continue
        if not (config.wd_qkv_max_prefetch_bytes or config.wu_q_max_prefetch_bytes or config.w_uk_t_max_prefetch_bytes):
            continue
        for index in range(backbone.start_layer, backbone.end_layer - 1):
            source = getattr(getattr(backbone.layers[index], "mlp", None), "experts", None)
            wrapper = getattr(getattr(backbone.layers[index + 1], "self_attn", None), "mla_attn", None)
            attention = getattr(wrapper, "mla_attn", None)
            if attention is None:
                continue
            target = getattr(attention, "impl", None)
            if not (
                isinstance(source, AscendMoERunner)
                and source.multistream_overlap_shared_expert
                and source._shared_experts is not None
                and source.quant_type == QuantType.W8A8
                and isinstance(target, AscendSFAKVOffloadImpl)
                and target.preprocess_type == PreprocessType.MLAPO
            ):
                continue
            weight = getattr(target, "wd_qkv", None)
            if weight is None or weight.device.type != "npu" or weight.numel() == 0 or weight.storage_offset() != 0:
                continue
            wu_q_weight = None
            if config.wu_q_max_prefetch_bytes > 0:
                wu_q_weight = getattr(target, "wu_q", None)
                if (
                    wu_q_weight is None
                    or wu_q_weight.device != weight.device
                    or wu_q_weight.numel() == 0
                    or wu_q_weight.storage_offset() != 0
                ):
                    logger.warning("Direct weight prefetch skipped for %s: invalid wu_q storage.", attention.layer_name)
                    continue
            w_uk_t_weight = None
            if config.w_uk_t_max_prefetch_bytes > 0:
                w_uk_t_weight = getattr(target, "W_UK_T", None)
                if (
                    w_uk_t_weight is None
                    or w_uk_t_weight.device != weight.device
                    or w_uk_t_weight.numel() == 0
                    or w_uk_t_weight.storage_offset() != 0
                    or not w_uk_t_weight.is_contiguous()
                ):
                    logger.warning(
                        "Direct weight prefetch skipped for %s: invalid W_UK_T storage.", attention.layer_name
                    )
                    continue
            if weight.device not in streams:
                streams[weight.device] = torch_npu.npu.Stream(device=weight.device)
            plan = DirectWeightPrefetchPlan(
                weight,
                attention.layer_name,
                streams[weight.device],
                config.wd_qkv_max_prefetch_bytes,
                wu_q_weight=wu_q_weight,
                wu_q_max_prefetch_bytes=config.wu_q_max_prefetch_bytes,
                w_uk_t_weight=w_uk_t_weight,
                w_uk_t_max_prefetch_bytes=config.w_uk_t_max_prefetch_bytes,
            )
            source._direct_weight_prefetch_out = plan
            target._direct_weight_prefetch_in = plan
            plans.append(plan)
    logger.info(
        "Direct weight prefetch bound %d adjacent-layer plans "
        "(wd_qkv limit=%d bytes, wu_q limit=%d bytes, W_UK_T limit=%d bytes; 0=disabled).",
        len(plans),
        config.wd_qkv_max_prefetch_bytes,
        config.wu_q_max_prefetch_bytes,
        config.w_uk_t_max_prefetch_bytes,
    )
    if config.o_proj_max_prefetch_bytes > 0:
        plans.extend(_bind_o_proj_weight_prefetch(model, config.o_proj_max_prefetch_bytes))
    return plans


def _bind_o_proj_weight_prefetch(model: torch.nn.Module, max_prefetch_bytes: int) -> list[OProjWeightPrefetchPlan]:
    # Binding is gated by the same target/offload execution checks as Direct.
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
    from vllm_ascend.attention.sfa_v1 import PreprocessType
    from vllm_ascend.quantization.methods import AscendW8A8LinearMethod

    plans = []
    streams: dict[torch.device, Any] = {}
    for backbone in model.modules():
        if not isinstance(backbone, DeepseekV2Model):
            continue
        # This is a same-layer plan, including dense/first/last local layers.
        for index in range(backbone.start_layer, backbone.end_layer):
            wrapper = getattr(getattr(backbone.layers[index], "self_attn", None), "mla_attn", None)
            attention = getattr(wrapper, "mla_attn", None)
            if attention is None:
                continue
            target = getattr(attention, "impl", None)
            if not (
                isinstance(target, AscendSFAKVOffloadImpl)
                and target.preprocess_type == PreprocessType.MLAPO
                and not getattr(target, "enable_dsa_cp_with_o_proj_tp", False)
            ):
                continue
            projection = getattr(target, "o_proj", None)
            if projection is None:
                continue
            adapter = getattr(projection, "quant_method", None)
            scheme = getattr(adapter, "quant_method", adapter)
            weight = getattr(projection, "weight", None)
            if (
                type(scheme) is not AscendW8A8LinearMethod
                or weight is None
                or weight.device.type != "npu"
                or weight.dtype != torch.int8
                or weight.numel() == 0
                or weight.storage_offset() != 0
            ):
                logger.warning(
                    "o_proj weight prefetch skipped for %s: unsupported weight or quantization.", attention.layer_name
                )
                continue
            if weight.device not in streams:
                streams[weight.device] = torch_npu.npu.Stream(device=weight.device)
            plan = OProjWeightPrefetchPlan(weight, attention.layer_name, streams[weight.device], max_prefetch_bytes)
            target._o_proj_weight_prefetch = projection._o_proj_weight_prefetch = plan
            plans.append(plan)
    logger.info("o_proj weight prefetch bound %d same-layer plans (limit=%d bytes).", len(plans), max_prefetch_bytes)
    return plans


class MTPRotWeightPrefetchPlan:
    """Eager draft logits -> sampling/next-step setup -> the next draft rot.

    This is separate from Direct plans: only draft decode may submit, and the
    proposer explicitly supplies whether another forward will consume it.
    No target-to-first-draft or cross-verification-round prefetch is issued.
    """

    def __init__(self, weight: torch.Tensor, stream: Any, max_prefetch_bytes: int):
        self.weight = weight
        self.stream = stream
        self.prefetch_bytes = min(weight.untyped_storage().nbytes(), max_prefetch_bytes)
        self.producer_ready = torch_npu.npu.Event()
        self.prefetch_done = torch_npu.npu.Event()
        self.submitted = False

    def submit(self, logits: torch.Tensor | None, *, has_next_draft: bool, is_prefill: bool | None) -> None:
        self.submitted = False
        if not has_next_draft or is_prefill is not False or logits is None or not is_forward_context_available():
            return
        context = get_forward_context()
        if (
            not getattr(context, "is_draft_model", False)
            or getattr(context, "in_profile_run", False)
            or getattr(context, "capturing", False)
            or logits.shape[0] == 0
            or logits.device != self.weight.device
        ):
            return
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            return
        if any(
            m is None
            or getattr(m, "attn_state", None)
            not in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            or getattr(m, "num_prefills", 0) != 0
            for m in metadata.values()
        ):
            return
        self.producer_ready.record(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(self.stream):
            self.stream.wait_event(self.producer_ready)
            torch_npu.npu_prefetch(self.weight, logits, self.prefetch_bytes)
            self.prefetch_done.record(self.stream)
            logits.record_stream(self.stream)
        self.submitted = True

    def wait(self) -> None:
        if self.submitted:
            torch_npu.npu.current_stream().wait_event(self.prefetch_done)
            self.submitted = False


def bind_mtp_rot_weight_prefetch(
    proposer: Any, vllm_config: "VllmConfig", config: "WeightPrefetchConfig"
) -> MTPRotWeightPrefetchPlan | None:
    """Bind the actual single-layer GLM rot storage after draft postprocessing."""
    if not config.enabled or config.mtp_rot_max_prefetch_bytes == 0:
        return None
    if not (
        proposer is not None
        and getattr(proposer, "method", None) == "mtp"
        and getattr(proposer, "num_speculative_tokens", 0) > 1
        and not getattr(proposer, "parallel_drafting", True)
        and not getattr(proposer, "use_cuda_graph", True)
        and getattr(proposer, "dcp_size", None) == 1
        and vllm_config.model_config.hf_text_config.model_type == "glm_moe_dsa"
        and not vllm_config.parallel_config.enable_dbo
        and vllm_config.lora_config is None
    ):
        logger.warning("MTP rot weight prefetch skipped: requires eager sequential GLM MTP without DCP/DBO/LoRA.")
        return None

    from vllm_ascend.patch.worker.patch_deepseek_mtp import (
        AscendDeepSeekMTP,
        AscendDeepSeekMultiTokenPredictorLayer,
    )

    model = getattr(proposer, "model", None)
    predictor = getattr(model, "model", None)
    if predictor is None or not isinstance(model, AscendDeepSeekMTP) or getattr(predictor, "num_mtp_layers", None) != 1:
        logger.warning("MTP rot weight prefetch skipped: requires the supported single-layer MTP model.")
        return None
    layer = predictor.layers[str(predictor.mtp_start_layer_idx)]
    rotation = getattr(layer, "rot", None)
    weight = getattr(rotation, "weight", None)
    if not (
        isinstance(layer, AscendDeepSeekMultiTokenPredictorLayer)
        and getattr(layer, "is_rot_used", False)
        and getattr(layer, "target_model_type", None) == "glm_moe_dsa"
        and isinstance(rotation, torch.nn.Linear)
        and weight is not None
        and weight.device.type == "npu"
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.shape[0] == weight.shape[1]
        and weight.numel() > 0
        and weight.storage_offset() == 0
        and weight.is_contiguous()
    ):
        logger.warning("MTP rot weight prefetch skipped: invalid BF16 rot storage or unsupported predictor layer.")
        return None
    if not callable(getattr(torch_npu, "npu_prefetch", None)):
        raise RuntimeError("weight_prefetch_config requires torch_npu.npu_prefetch")
    plan = MTPRotWeightPrefetchPlan(
        weight, torch_npu.npu.Stream(device=weight.device), config.mtp_rot_max_prefetch_bytes
    )
    proposer._mtp_rot_weight_prefetch = layer._mtp_rot_weight_prefetch = plan
    logger.info(
        "MTP rot weight prefetch bound 1 eager plan (storage=%d bytes, limit=%d bytes, actual=%d bytes; "
        "draft logits -> next rot, last step skipped).",
        weight.untyped_storage().nbytes(),
        config.mtp_rot_max_prefetch_bytes,
        plan.prefetch_bytes,
    )
    return plan
