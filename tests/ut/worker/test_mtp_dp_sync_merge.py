"""CPU-only checks of the production merge protocol and cache lifetime.

Run directly without importing vLLM/NPU: python test_mtp_dp_sync_merge.py
Real Linux collectives: python test_mtp_dp_sync_merge.py --gloo \
    --ranks-per-group 3 --groups 2
Full-call microbenchmark: torchrun <rendezvous arguments> --nproc-per-node=16 \
    test_mtp_dp_sync_merge.py --gloo --torchrun --benchmark --groups 2 \
    --group-layout interleaved --output /absolute/path/mtp-dp-sync.json
The --gloo mode requires a CPU-capable torch build and never initializes NPU.
"""

import ast
import copy
import hashlib
import json
import os
import platform
import runpy
import socket
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from enum import IntEnum
from functools import partial, wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/worker/model_runner_v1.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
CONTROL_SOURCES = {
    ROOT / "vllm_ascend/utils.py": {"should_skip_allreduce_across_dp_group", "lmhead_tp_enable"},
    ROOT / "vllm_ascend/ascend_config.py": {"get_ascend_config", "_is_ascend_config_initialized"},
}


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Tensor(np.ndarray):
    def cpu(self):
        return self


class CpuArrays:
    """NumPy storage adapter; the synchronization logic comes from source AST."""
    int32 = np.int32

    @staticmethod
    def zeros(*shape, device=None, dtype=None):
        return np.zeros(shape, dtype=dtype).view(Tensor)

    @staticmethod
    def tensor(value, device=None, dtype=None):
        return np.array(value, dtype=dtype).view(Tensor)

    @staticmethod
    def full(shape, value, device=None, dtype=None):
        return np.full(shape, value, dtype=dtype).view(Tensor)

    where = staticmethod(np.where)

    @staticmethod
    def inference_mode():
        return lambda method: method


class Proposer:
    use_cuda_graph = False
    needs_extra_input_slots = False


def load_production(torch=CpuArrays, merge=True):
    methods = {"_sync_metadata_across_dp", "_can_merge_mtp_dp_sync", "_mtp_dp_sync_active_role",
               "execute_model", "sample_tokens", "_dummy_run",
               "_determine_batch_execution_and_padding", "_pad_for_sequence_parallelism"}
    cls = copy.deepcopy(next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner"))
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(copy.deepcopy(n) for n in TREE.body if (
        isinstance(n, ast.FunctionDef) and n.name in {"_dp_sync_lifetime", "_post_process_cudagraph_mode"}
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_MTP_DP_DUMMY_")
                                           for t in n.targets)))
    body.append(cls)
    ns = dict(torch=torch, np=np, partial=partial, wraps=wraps, _MERGE_MTP_DP_SYNC=merge,
              CUDAGraphMode=Mode, AscendEagleProposer=Proposer,
              should_skip_allreduce_across_dp_group=lambda *a: False,
              get_dp_group=lambda: SimpleNamespace(cpu_group="cpu-group"),
              get_pp_group=lambda: SimpleNamespace(world_size=1),
              EMPTY_MODEL_RUNNER_OUTPUT=object())
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns


def runner(ns, rank=0, size=3):
    r = ns["NPUModelRunner"].__new__(ns["NPUModelRunner"])
    r.dp_rank, r.dp_size, r.dcp_size = rank, size, 1
    r.parallel_config = SimpleNamespace(pipeline_parallel_size=1, enable_dbo=False)
    r.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False))
    r.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(
        profiling_chunk_config=SimpleNamespace(need_timing=False)))
    r.speculative_config = SimpleNamespace(method="mtp", disable_padded_drafter_batch=False)
    # Ascend resets disable_cascade_attn=False before GPUModelRunner init.
    r.cascade_attn_enabled = True
    r.drafter = Proposer()
    r.input_batch = SimpleNamespace(num_reqs=1, lora_id_to_lora_request={},
                                   num_computed_tokens_cpu=np.array([100]), num_prompt_tokens=np.array([100]))
    r.uniform_decode_query_len = 6
    r.execute_model_state = None
    r.kv_connector_output = None
    r._mtp_dp_sync_role = 0
    r._merged_mtp_dp_sync = None
    return r


class Collective:
    def __init__(self, matrices):
        self.matrices = list(matrices)
        self.calls = []
        self.events = []

    def all_reduce(self, value, *, group, async_op=False):
        self.calls.append((value.copy(), group))
        self.events.append("submit" if async_op else "sync")
        result = self.matrices.pop(0)
        if async_op:
            def wait():
                self.events.append("wait")
                value[...] = result

            return SimpleNamespace(wait=wait)
        value[...] = result


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.p = load_production()

    def test_main_and_first_draft_match_original_for_all_roles(self):
        for rank in range(3):
            with self.subTest(rank=rank):
                fake = Collective([[[8, 4, 12], [2, 0, 1], [6, -1, -2]],
                                   [[6, 4, 12], [0, 0, 0]]])
                self.p["dist"] = fake
                r = runner(self.p, rank)
                r._mtp_dp_sync_role = [6, -1, -1][rank]
                main = r._sync_metadata_across_dp([8, 4, 12][rank],
                    cudagraph_mode=[Mode.FULL, Mode.NONE, Mode.PIECEWISE][rank],
                    allow_dp_padding=rank != 1)
                self.assertEqual(fake.calls[0][0][2, rank], [6, -1, -2][rank])
                self.assertEqual(main[1].tolist(), [12] * 3 if rank != 1 else [8, 4, 12])
                draft = r._sync_metadata_across_dp([6, 4, 12][rank], is_draft_model=True)
                self.assertEqual((draft[0], draft[1].tolist(), draft[2]), (12, [12] * 3, Mode.NONE))
                self.assertEqual(len(fake.calls), 1)
                self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
                r._sync_metadata_across_dp([6, 4, 12][rank], is_draft_model=True)
                self.assertEqual(len(fake.calls), 2, "a second draft call must not reuse the prediction")

    def test_any_ineligible_rank_keeps_both_collectives_symmetrically(self):
        for role in (0, -3):
            for rank in range(3):
                with self.subTest(role=role, rank=rank):
                    fake = Collective([[[8, 4, 12], [2, 0, 1], [6, role, -2]],
                                       [[6, 4, 12], [0, 0, 0]]])
                    self.p["dist"] = fake
                    r = runner(self.p, rank)
                    r._sync_metadata_across_dp([8, 4, 12][rank])
                    self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
                    r._sync_metadata_across_dp([6, 4, 12][rank], is_draft_model=True)
                    self.assertEqual(len(fake.calls), 2)

    def test_prediction_survives_mutation_of_unpadded_main_token_row(self):
        self.p["dist"] = fake = Collective([[[8, 4, 12], [0, 0, 0], [6, -1, -2]]])
        r = runner(self.p, rank=1)
        r._mtp_dp_sync_role = -1
        _, main_counts, _ = r._sync_metadata_across_dp(4)
        main_counts[:] = 19
        draft_max, draft_counts, draft_mode = r._sync_metadata_across_dp(4, is_draft_model=True)
        self.assertEqual((draft_max, draft_counts.tolist(), draft_mode), (12, [12] * 3, Mode.NONE))
        self.assertEqual(draft_counts.dtype, np.int32)
        self.assertEqual(len(fake.calls), 1)

    def test_invalid_prediction_raises_without_extra_collective(self):
        for tokens, mode in ((5, Mode.NONE), (6, Mode.FULL)):
            with self.subTest(tokens=tokens, mode=mode):
                self.p["dist"] = Collective([])
                r = runner(self.p)
                r._merged_mtp_dp_sync = (6, (12, None, Mode.NONE))
                with self.assertRaisesRegex(RuntimeError, "prediction"):
                    r._sync_metadata_across_dp(tokens, is_draft_model=True, cudagraph_mode=mode)
                self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
                self.assertFalse(self.p["dist"].calls)

    def test_disabled_path_preserves_original_tensor_and_collectives(self):
        p = load_production(merge=False)
        fake = Collective([[[2, 7, 4], [2, 1, 2]], [[2, 7, 4], [0, 0, 0]]])
        p["dist"] = fake
        r = runner(p)
        main = r._sync_metadata_across_dp(2, cudagraph_mode=Mode.FULL)
        draft = r._sync_metadata_across_dp(2, is_draft_model=True)
        self.assertEqual((main[0], main[1].tolist(), main[2]), (7, [2, 7, 4], Mode.PIECEWISE))
        self.assertEqual((draft[0], draft[1].tolist(), draft[2]), (7, [7, 7, 7], Mode.NONE))
        self.assertTrue(main[1].flags.c_contiguous)
        self.assertEqual(main[1].dtype, np.int32)
        self.assertEqual([c[0].shape for c in fake.calls], [(2, 3), (2, 3)])
        method = lambda self: None
        self.assertIs(p["_dp_sync_lifetime"](new_step=True)(method), method)

    def test_short_tail_first_step_end_and_non_decode(self):
        r = runner(self.p)
        out = SimpleNamespace(total_num_scheduled_tokens=6, scheduled_encoder_inputs={})
        self.assertEqual(r._mtp_dp_sync_active_role(out, [6]), 6)
        out.total_num_scheduled_tokens = 1
        self.assertEqual(r._mtp_dp_sync_active_role(out, [1]), 1)
        for tokens in ([], [0], [-1], [7], [1, 1]):
            self.assertEqual(r._mtp_dp_sync_active_role(out, tokens), 0)
        out.total_num_scheduled_tokens = 0
        self.assertEqual(r._mtp_dp_sync_active_role(out, []), 0)
        out.total_num_scheduled_tokens = 1
        out.scheduled_encoder_inputs = {"encoder": 1}
        self.assertEqual(r._mtp_dp_sync_active_role(out, [1]), 0)
        out.scheduled_encoder_inputs = {}
        r.input_batch.num_computed_tokens_cpu[0] = 99
        self.assertEqual(r._mtp_dp_sync_active_role(out, [1]), 0)

    def test_dynamic_dp_sizes_and_multiple_requests(self):
        for size in (2, 3, 7, 16):
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    r = runner(self.p, rank, size)
                    self.assertTrue(r._can_merge_mtp_dp_sync())
                    counts = list(range(1, size + 1))
                    self.p["dist"] = Collective([[counts, [0] * size, counts]])
                    r._mtp_dp_sync_role = counts[rank]
                    r._sync_metadata_across_dp(counts[rank])
                    result = r._sync_metadata_across_dp(counts[rank], is_draft_model=True)
                    self.assertEqual(result[1].tolist(), [size] * size)
        r = runner(self.p)
        r.input_batch.num_reqs = 2
        r.input_batch.num_computed_tokens_cpu = np.array([100, 200])
        r.input_batch.num_prompt_tokens = np.array([100, 200])
        out = SimpleNamespace(total_num_scheduled_tokens=5, scheduled_encoder_inputs={})
        self.assertEqual(r._mtp_dp_sync_active_role(out, [2, 3]), 5)

    def test_cascade_flag_and_graph_padding_preserve_first_draft_counts(self):
        # Run the production dispatch -> sync -> redispatch path. The dispatcher
        # models the formal main graph size 6; the draft remains eager. In
        # particular, a dummy's raw 1/2 tokens have already become 6 before sync.
        for cascade in (False, True):
            for active_tokens in (1, 6, 12):
                for rank in range(3):
                    with self.subTest(cascade=cascade, active_tokens=active_tokens, rank=rank):
                        _, p, r = benchmark_pair(CpuArrays, Mock(), "cpu-group", rank, 3, "all_active", 1, True)
                        p.update(enable_sp=lambda *a: False, enable_sp_by_pass=lambda: False,
                                 oproj_tp_enable=lambda: False, embedding_tp_enable=lambda: False,
                                 BatchDescriptor=lambda n: SimpleNamespace(num_tokens=n, num_reqs=None))
                        r.cascade_attn_enabled = cascade
                        r.model_config = SimpleNamespace(is_encoder_decoder=False)
                        r.parallel_config.data_parallel_size = 3
                        r.parallel_config.data_parallel_rank = rank
                        r.parallel_config.tensor_parallel_size = 2
                        r.vllm_config.parallel_config = r.parallel_config
                        r.vllm_config.observability_config = SimpleNamespace(cudagraph_metrics=False)
                        dispatched = []

                        def dispatch(num_tokens, *, valid_modes=None, invalid_modes=None, **kwargs):
                            mode = (Mode.FULL if num_tokens <= 6 and Mode.FULL not in (invalid_modes or ())
                                    and (valid_modes is None or Mode.FULL in valid_modes) else Mode.NONE)
                            count = 6 if mode == Mode.FULL else num_tokens
                            dispatched.append((num_tokens, valid_modes, count, invalid_modes))
                            return mode, SimpleNamespace(num_tokens=count, num_reqs=None)

                        r.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)
                        out = SimpleNamespace(total_num_scheduled_tokens=active_tokens, scheduled_encoder_inputs={})
                        r._mtp_dp_sync_role = r._mtp_dp_sync_active_role(out, [active_tokens]) if rank == 0 else -1
                        active_uses_full_graph = active_tokens <= 6 and not cascade
                        main_counts = [6 if active_uses_full_graph else active_tokens, 6, 6]
                        roles = [active_tokens if active_tokens <= 6 else 0, -2, -2]
                        draft_counts = [active_tokens, max(main_counts), max(main_counts)]
                        matrices = [[main_counts, [2 if active_uses_full_graph else 0, 2, 2], roles]]
                        if active_tokens > 6:
                            matrices.append([draft_counts, [0, 0, 0]])
                        p["dist"] = fake = Collective(matrices)
                        overlap = rank == 0 and r._mtp_dp_sync_role > 0

                        def prepare():
                            self.assertEqual(fake.events, ["submit"])
                            self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
                            fake.events.append("prepare")

                        _, desc, _, across_dp, _ = r._determine_batch_execution_and_padding(
                            num_tokens=active_tokens if rank == 0 else rank, num_reqs=1,
                            num_scheduled_tokens_np=np.array([active_tokens if rank == 0 else rank]),
                            max_num_scheduled_tokens=active_tokens if rank == 0 else rank,
                            use_cascade_attn=cascade and rank == 0, force_uniform_decode=rank != 0,
                            prepare_inputs_during_sync=prepare if overlap else None)
                        self.assertEqual(dispatched[0][3], {Mode.FULL} if cascade and rank == 0 else None)
                        self.assertEqual(fake.events, ["submit", "prepare", "wait"] if overlap else ["sync"])
                        self.assertEqual(fake.calls[0][0][0, rank], main_counts[rank])
                        self.assertEqual(fake.calls[0][0][2, rank], roles[rank])
                        self.assertEqual(desc.num_tokens, max(main_counts))
                        self.assertEqual(across_dp.tolist(), [max(main_counts)] * 3)
                        self.assertEqual(dispatched[1][0], desc.num_tokens)
                        # _dummy_run passes this exact final batch size to its
                        # proposer; active padded drafting retains local input.
                        draft_tokens = active_tokens if rank == 0 else desc.num_tokens
                        result = r._sync_metadata_across_dp(draft_tokens, is_draft_model=True)
                        self.assertEqual(result[0], max(draft_counts))
                        self.assertEqual(result[1].tolist(), [max(draft_counts)] * 3)
                        self.assertEqual(len(fake.calls), 2 if active_tokens > 6 else 1)

    def test_early_collective_drains_if_input_preparation_raises(self):
        r = runner(self.p)
        matrix = [[8, 4, 12], [2, 0, 1], [6, -1, -2]]
        self.p["dist"] = fake = Collective([matrix, matrix])

        def prepare():
            self.assertEqual(fake.events, ["submit"])
            self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
            fake.events.append("prepare")
            raise ValueError("prepare failed")

        @self.p["_dp_sync_lifetime"](new_step=True, keep_for_sample=True)
        def execute(self):
            self._mtp_dp_sync_role = 6
            return self._sync_metadata_across_dp(
                8, cudagraph_mode=Mode.FULL, allow_dp_padding=True, prepare_inputs_during_sync=prepare)

        with self.assertRaisesRegex(ValueError, "prepare failed"):
            execute(r)
        self.assertEqual(fake.events, ["submit", "prepare", "wait"])
        self.assertEqual(len(fake.calls), 1)
        self.assertIsNone(r._merged_mtp_dp_sync)
        self.assertEqual(r._mtp_dp_sync_role, 0)

        # A fresh step can reuse the same group after draining the failed step.
        r._mtp_dp_sync_role = 6
        main = r._sync_metadata_across_dp(8, cudagraph_mode=Mode.FULL, allow_dp_padding=True)
        draft = r._sync_metadata_across_dp(6, is_draft_model=True)
        self.assertEqual(main[1].tolist(), [12] * 3)
        self.assertEqual(draft[1].tolist(), [12] * 3)
        self.assertEqual(fake.events, ["submit", "prepare", "wait", "sync"])

    def test_actual_unsupported_paths_do_not_publish_predictions(self):
        cases = [("dp_size", 1), ("dcp_size", 2), ("parallel_config.pipeline_parallel_size", 2),
                 ("parallel_config.enable_dbo", True),
                 ("speculative_config", None), ("speculative_config.method", "eagle"),
                 ("speculative_config.disable_padded_drafter_batch", True), ("drafter", None),
                 ("drafter.use_cuda_graph", True), ("drafter.needs_extra_input_slots", True),
                 ("input_batch.lora_id_to_lora_request", {1: 1})]
        for path, value in cases:
            with self.subTest(path=path):
                r = runner(self.p)
                target = r
                fields = path.split(".")
                for name in fields[:-1]:
                    target = getattr(target, name)
                setattr(target, fields[-1], value)
                self.assertFalse(r._can_merge_mtp_dp_sync())
        for phase in (False, True):
            self.p["should_skip_allreduce_across_dp_group"] = lambda cfg, draft: draft == phase
            self.assertFalse(runner(self.p)._can_merge_mtp_dp_sync())

    def test_dummy_capture_profile_and_non_decode(self):
        cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run")
        marker = next(n for n in method.body if isinstance(n, ast.If) and any(
            isinstance(x, ast.Attribute) and x.attr == "_mtp_dp_sync_role" for x in ast.walk(n)))
        code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(marker)], type_ignores=[])),
                       str(SOURCE), "exec")
        defaults = dict(uniform_decode=True, num_active_loras=0, cudagraph_runtime_mode=None,
                        with_prefill=False, is_profile=False, profile_cpp=False,
                        is_graph_capturing=False, create_mixed_batch=False)
        for key in (None, *defaults):
            with self.subTest(key=key):
                r = runner(self.p)
                args = defaults.copy()
                if key:
                    args[key] = False if key == "uniform_decode" else True
                exec(code, dict(self.p, self=r, **args))
                self.assertEqual(r._mtp_dp_sync_role, -1 if key is None else 0)

    def test_execute_sample_lifetime_and_exceptions(self):
        r = runner(self.p)
        lifetime = self.p["_dp_sync_lifetime"]

        @lifetime(new_step=True, keep_for_sample=True)
        def execute(self, outcome="normal"):
            self.assertion = self._merged_mtp_dp_sync is None and self._mtp_dp_sync_role == 0
            self._merged_mtp_dp_sync = (6, (8, CpuArrays.tensor([8] * 3), Mode.NONE))
            self._mtp_dp_sync_role = 6
            if outcome == "fail":
                raise ValueError("prepare failed")
            if outcome != "early":
                self.execute_model_state = object()

        @lifetime(new_step=False)
        def sample(self, fail=False):
            self.execute_model_state = None
            if fail:
                raise ValueError("sample failed")
            return self._sync_metadata_across_dp(6, is_draft_model=True)

        for outcome in ("normal", "early", "fail"):
            r.execute_model_state = None
            r._merged_mtp_dp_sync = object()
            if outcome == "fail":
                with self.assertRaises(ValueError):
                    execute(r, outcome)
            else:
                execute(r, outcome)
            self.assertTrue(r.assertion)
            self.assertEqual(r._mtp_dp_sync_role, 0)
            if outcome == "normal":
                self.assertIsNotNone(r._merged_mtp_dp_sync)
                self.assertEqual(sample(r)[0], 8)
            self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
        execute(r)
        with self.assertRaisesRegex(ValueError, "sample failed"):
            sample(r, True)
        self.assertIsNone(r._merged_mtp_dp_sync)

    def test_real_method_entries_clear_stale_cache_on_errors_and_early_return(self):
        r = runner(self.p)
        r._merged_mtp_dp_sync = object()
        self.assertIsNone(r.sample_tokens(None))
        self.assertIsNone(r._merged_mtp_dp_sync)
        r._merged_mtp_dp_sync = object()
        self.p["get_pp_group"] = Mock(side_effect=ValueError("pp failure"))
        with self.assertRaises(ValueError):
            r.sample_tokens(None)
        self.assertIsNone(r._merged_mtp_dp_sync)
        r._merged_mtp_dp_sync = object()
        r.execute_model_state = object()
        with self.assertRaisesRegex(RuntimeError, "State error"):
            r.execute_model(SimpleNamespace())
        self.assertIsNone(r._merged_mtp_dp_sync)
        self.assertEqual(r._mtp_dp_sync_role, 0)
        r._merged_mtp_dp_sync = object()
        with self.assertRaises(AssertionError):
            r._dummy_run(1, cudagraph_runtime_mode=SimpleNamespace(valid_runtime_modes=lambda: False))
        self.assertIsNone(r._merged_mtp_dp_sync)

    def test_dummy_exit_and_collective_failure_clear_predictions(self):
        r = runner(self.p)

        @self.p["_dp_sync_lifetime"](new_step=True)
        def dummy(self):
            self._merged_mtp_dp_sync = object()
            self.execute_model_state = object()

        dummy(r)
        self.assertIsNone(r._merged_mtp_dp_sync)
        self.p["dist"] = SimpleNamespace(all_reduce=Mock(side_effect=RuntimeError("sync failed")))
        r._merged_mtp_dp_sync = object()
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            r._sync_metadata_across_dp(6)
        self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))

    def test_main_fast_paths_clear_previous_prediction(self):
        for size, skip in ((1, False), (3, True)):
            r = runner(self.p, size=size)
            r._merged_mtp_dp_sync = object()
            self.p["should_skip_allreduce_across_dp_group"] = lambda *a: skip
            result = r._sync_metadata_across_dp(6)
            self.assertEqual(result[0], 6)
            self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))

    def test_flag_defaults_to_zero(self):
        envs = runpy.run_path(str(ROOT / "vllm_ascend/envs.py"))["env_variables"]
        name = "VLLM_ASCEND_MERGE_MTP_DP_SYNC"
        with patch.dict(os.environ):
            os.environ.pop(name, None)
            self.assertFalse(envs[name]())
            os.environ[name] = "1"
            self.assertTrue(envs[name]())

    def test_benchmark_prepares_real_roles_and_reuses_step_lifetimes(self):
        self.assertEqual(group_members(16, 2, "interleaved"), [list(range(0, 32, 2)), list(range(1, 32, 2))])
        self.assertEqual(group_members(2, 3, "contiguous"), [[0, 1], [2, 3], [4, 5]])
        for scenario in ("all_active", "mixed", "short_tail", "all_dummy", "ineligible"):
            counts = [2, 2, 2] if scenario == "short_tail" else [2, 4, 6]
            roles = counts.copy()
            if scenario in ("mixed", "short_tail"):
                roles[1:] = [-1, -2]
            elif scenario == "all_dummy":
                roles = [-1, -1, -2]
            elif scenario == "ineligible":
                roles[0] = 0
            drafts = [max(counts) if role == -2 else count if role == -1 else role or count
                      for role, count in zip(roles, counts)]
            for rank in range(3):
                results = []
                for enabled in (False, True):
                    with self.subTest(scenario=scenario, rank=rank, enabled=enabled):
                        matrices = [[counts, [0] * 3] + ([roles] if enabled else [])]
                        if not enabled or scenario == "ineligible":
                            matrices.append([drafts, [0] * 3])
                        fake = Collective(copy.deepcopy(matrices) * 2)
                        pair, _, r = benchmark_pair(CpuArrays, fake, "cpu-group", rank, 3, scenario, 2, enabled)
                        first = pair()
                        second = pair()
                        as_values = lambda result: [(x[0], x[1].tolist(), x[2]) for x in result]
                        self.assertEqual(as_values(first), as_values(second))
                        self.assertEqual(len(fake.calls), 2 * len(matrices))
                        self.assertIsNone(r.__dict__.get("_merged_mtp_dp_sync"))
                        results.append(as_values(first))
                self.assertEqual(results[0], results[1])


def group_members(ranks_per_group, groups, layout):
    world_size = ranks_per_group * groups
    if layout == "interleaved":
        return [list(range(i, world_size, groups)) for i in range(groups)]
    return [list(range(i * ranks_per_group, (i + 1) * ranks_per_group)) for i in range(groups)]


def benchmark_pair(torch, dist, group, rank, size, scenario, num_reqs, enabled):
    """Real production sync calls, including role preparation and cache lifetime."""
    p = load_production(torch=torch, merge=enabled)
    p["dist"] = dist
    dp_group = SimpleNamespace(cpu_group=group)
    p["get_dp_group"] = lambda: dp_group
    r = runner(p, rank, size)
    # This isolated benchmark uses the formal run's LMHead TP16 / KV-consumer
    # config. Group membership and group size remain independent parameters.
    # Execute the real helper's LMHead-TP branch, including config validation,
    # rather than replacing each extra eligibility check with lambda: False.
    r.ascend_config.finegrained_tp_config = SimpleNamespace(lmhead_tensor_parallel_size=16)
    r.ascend_config.ascend_compilation_config = None
    r.ascend_config.eplb_config = None
    r.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    p["_ASCEND_CONFIG"] = r.ascend_config
    for source, names in CONTROL_SOURCES.items():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        body.extend(copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in names)
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), p)
    r.input_batch.num_reqs = num_reqs
    r.input_batch.num_computed_tokens_cpu = np.full(num_reqs, 100, dtype=np.int32)
    r.input_batch.num_prompt_tokens = np.full(num_reqs, 100, dtype=np.int32)
    query_len = 1 if scenario == "short_tail" else 1 + rank % r.uniform_decode_query_len
    tokens = [query_len] * num_reqs
    main_count = sum(tokens)
    max_count = num_reqs * (1 if scenario == "short_tail" else min(size, r.uniform_decode_query_len))
    dummy = scenario == "all_dummy" or scenario in ("mixed", "short_tail") and rank % 3 != 0
    padded = dummy and rank % 3 == 2
    draft_count = max_count if padded else main_count
    out = SimpleNamespace(total_num_scheduled_tokens=main_count, scheduled_encoder_inputs={})
    if scenario == "ineligible" and rank == 0:
        r.drafter.use_cuda_graph = True

    # Compile the actual dummy-role marker as a small callable. Its default
    # arguments are the ordinary eager uniform-decode dummy-run inputs.
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run")
    marker = next(n for n in method.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Attribute) and x.attr == "_mtp_dp_sync_role" for x in ast.walk(n)))
    prepare = ast.parse("""def prepare_dummy(self, uniform_decode=True, num_active_loras=0,
            cudagraph_runtime_mode=None, with_prefill=False, is_profile=False,
            profile_cpp=False, is_graph_capturing=False, create_mixed_batch=False):
        pass
""").body[0]
    prepare.body = [copy.deepcopy(marker)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prepare], type_ignores=[])),
                 str(SOURCE), "exec"), p)

    @p["_dp_sync_lifetime"](new_step=True, keep_for_sample=True)
    def main(self):
        if p["_MERGE_MTP_DP_SYNC"]:
            self._mtp_dp_sync_role = self._mtp_dp_sync_active_role(out, tokens)
        result = self._sync_metadata_across_dp(main_count, cudagraph_mode=Mode.NONE)
        self.execute_model_state = True
        return result

    @p["_dp_sync_lifetime"](new_step=False)
    def draft(self):
        self.execute_model_state = None
        return self._sync_metadata_across_dp(draft_count, is_draft_model=True)

    @p["_dp_sync_lifetime"](new_step=True)
    def dummy_pair(self):
        p["prepare_dummy"](self)
        return (self._sync_metadata_across_dp(main_count, cudagraph_mode=Mode.NONE, allow_dp_padding=padded),
                self._sync_metadata_across_dp(draft_count, is_draft_model=True))

    def pair():
        return dummy_pair(r) if dummy else (main(r), draft(r))

    pair.batch = dict(main_count=main_count, draft_count=draft_count, tokens=tokens, dummy=dummy, padded=padded)
    return pair, p, r


def gloo_benchmark(torch, dist, group, global_rank, rank, members, options, counted_reduce):
    pairs = {}
    results = []
    for enabled in (False, True):
        pair, p, r = benchmark_pair(torch, dist, group, rank, len(members), options.scenario,
                                    options.requests_per_rank, enabled)
        # One untimed preflight also verifies that production role preparation
        # actually permits the one-collective path for the selected batch.
        calls = 0

        def check_reduce(value, *, group):
            nonlocal calls
            calls += 1
            counted_reduce(value, group=group)

        p["dist"] = SimpleNamespace(all_reduce=check_reduce)
        result = pair()
        expected = 1 if enabled and options.scenario != "ineligible" else 2
        assert calls == expected, (options.scenario, global_rank, enabled, calls)
        assert r.__dict__.get("_merged_mtp_dp_sync") is None
        results.append([(x[0], x[1].tolist(), int(x[2])) for x in result])
        p["dist"] = dist  # No collective wrapper or telemetry in the timed calls.
        pairs["B" if enabled else "A"] = pair
    assert results[0] == results[1], (global_rank, results)

    dist.barrier(group=group)
    for _ in range(options.warmup):
        pairs["A"]()
        pairs["B"]()
    samples = []
    for block in range(options.blocks):
        order = "ABBA" if block % 2 == 0 else "BAAB"
        for arm, variant in enumerate(order):
            pair = pairs[variant]
            for sample in range(options.samples_per_arm):
                # Synchronize arrivals without counting this benchmark-only
                # barrier in the main+draft metadata latency.
                dist.barrier(group=group)
                start = time.perf_counter_ns()
                pair()
                elapsed = time.perf_counter_ns() - start
                samples.append(dict(block=block, order=order, arm=arm, sample=sample,
                                    variant=variant, elapsed_ns=elapsed))
    return dict(global_rank=global_rank, rank_in_group=rank, members=members,
                hostname=socket.gethostname(), local_rank=os.environ.get("LOCAL_RANK"),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                batch=pairs["B"].batch, samples=samples)


def write_benchmark(records, memberships, options, torch):
    def statistics(samples):
        return dict(count=len(samples), mean_ns=float(np.mean(samples)),
                    median_ns=float(np.median(samples)), p95_ns=float(np.percentile(samples, 95)),
                    min_ns=min(samples), max_ns=max(samples))

    grouped = []
    for members in memberships:
        ranks = [record for record in records if record["global_rank"] in members]
        maxima = []
        for samples in zip(*(record["samples"] for record in ranks)):
            key = {k: v for k, v in samples[0].items() if k != "elapsed_ns"}
            assert all({k: v for k, v in sample.items() if k != "elapsed_ns"} == key for sample in samples)
            maxima.append(dict(key, elapsed_ns=max(sample["elapsed_ns"] for sample in samples)))
        summary = {variant: statistics([sample["elapsed_ns"] for sample in maxima
                                        if sample["variant"] == variant]) for variant in ("A", "B")}
        grouped.append(dict(members=members, group_max_samples=maxima, summary=summary,
                            speedup=summary["A"]["mean_ns"] / summary["B"]["mean_ns"]))
    report = dict(schema_version=1, backend="gloo", torch_version=torch.__version__,
                  python_version=platform.python_version(), numpy_version=np.__version__,
                  source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  control_source_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                                         for source in CONTROL_SOURCES},
                  world_size=len(records), memberships=memberships, scenario=options.scenario,
                  requests_per_rank=options.requests_per_rank, warmup_pairs_per_variant=options.warmup,
                  blocks=options.blocks, samples_per_arm=options.samples_per_arm,
                  variants={"A": "original main+draft (two metadata all-reduces)",
                            "B": "merged main+cached draft (one metadata all-reduce); ineligible uses two"},
                  timed_scope="production role preparation + lifecycle wrappers + production main/draft sync calls",
                  excluded_scope="runner/input setup, warmup, barriers, correctness assertions and result aggregation",
                  control_config=dict(lmhead_tensor_parallel_size=16, is_kv_consumer=True,
                                      branch="real should_skip_allreduce_across_dp_group LMHead-TP branch"),
                  scope_limit="isolated CPU metadata path with minimal runner state; no model, NPU or full inference timing",
                  environment={name: os.environ.get(name) for name in
                               ("GLOO_SOCKET_IFNAME", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  per_rank=records, groups=grouped)
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for i, result in enumerate(grouped):
        summary = result["summary"]
        print(f"group {i}: A={summary['A']['mean_ns'] / 1000:.3f} us, "
              f"B={summary['B']['mean_ns'] / 1000:.3f} us, "
              f"speedup={result['speedup']:.4f} (mean of per-sample group maxima)", flush=True)
    print(f"Saved per-rank samples and group maxima: {output}", flush=True)


def verify_gloo_overlap(torch, dist, dp_group, rank, size):
    p = load_production(torch=torch)
    p["get_dp_group"] = lambda: SimpleNamespace(cpu_group=dp_group)
    active = rank == 0
    local_tokens = 4 + 2 * rank
    draft_tokens = 3 if active else local_tokens

    def values(result):
        maximum, counts, mode = result
        return maximum, counts.tolist(), mode

    def run_trial(*, overlap, fail_preparation=False):
        r = runner(p, rank, size)
        payloads, events = [], []
        failure = ValueError("injected active input preparation failure")

        def record_reduce(value, *, group, async_op=False):
            assert group is dp_group
            assert value.dtype == torch.int32 and value.device.type == "cpu"
            payloads.append(value.tolist())
            events.append("submit" if async_op else "sync")
            work = dist.all_reduce(value, group=group, async_op=async_op)
            if not async_op:
                return None

            def wait():
                result = work.wait()
                events.append("wait")
                return result

            return SimpleNamespace(wait=wait)

        def prepare():
            assert events == ["submit"]
            assert r.__dict__.get("_merged_mtp_dp_sync") is None
            events.append("prepare")
            if fail_preparation:
                raise failure

        @p["_dp_sync_lifetime"](new_step=True, keep_for_sample=True)
        def first_step(self):
            self._mtp_dp_sync_role = draft_tokens if active else -1
            output = self._sync_metadata_across_dp(
                local_tokens, cudagraph_mode=Mode.FULL if active else Mode.NONE,
                allow_dp_padding=active, prepare_inputs_during_sync=prepare if overlap and active else None,
            )
            self.execute_model_state = object()
            return output

        p["dist"] = SimpleNamespace(all_reduce=record_reduce)
        try:
            first = values(first_step(r))
        except ValueError as exc:
            assert overlap and active and fail_preparation and exc is failure
            assert r.__dict__.get("_merged_mtp_dp_sync") is None and r._mtp_dp_sync_role == 0
            first, draft = None, None
        else:
            assert not (fail_preparation and active)
            draft = values(r._sync_metadata_across_dp(draft_tokens, is_draft_model=True))

        assert len(payloads) == 1
        assert events == (["submit", "prepare", "wait"] if overlap and active else ["sync"])
        # Use a different payload for the following main collective so a
        # mismatched sequence cannot accidentally return the reference values.
        following = values(r._sync_metadata_across_dp(local_tokens + 17))
        assert len(payloads) == 2 and events[-1] == "sync"
        return first, draft, following, payloads

    synchronous = run_trial(overlap=False)
    assert run_trial(overlap=True) == synchronous
    after_failure = run_trial(overlap=True, fail_preparation=True)
    assert after_failure[2:] == synchronous[2:]
    assert after_failure[:2] == ((None, None) if active else synchronous[:2])


def gloo_worker(global_rank, ranks_per_group, groups, store, options=None):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store, rank=global_rank,
                            world_size=ranks_per_group * groups, timeout=timedelta(seconds=90))
    try:
        memberships = group_members(ranks_per_group, groups, options.group_layout if options else "contiguous")
        group = None
        own_members = None
        for members in memberships:
            created = dist.new_group(members, backend="gloo")
            if global_rank in members:
                group = created
                own_members = members
        rank = own_members.index(global_rank)
        p = load_production(torch=torch)
        calls = 0

        def counted_reduce(value, *, group):
            nonlocal calls
            calls += 1
            dist.all_reduce(value, group=group)

        p["dist"] = SimpleNamespace(all_reduce=counted_reduce)
        p["get_dp_group"] = lambda: SimpleNamespace(cpu_group=group)
        # Mixed real/dummy roles and a short tail, all dummy, and one rank
        # refusing the merge. Every process executes the same phase ordering.
        for scenario in ("mixed", "short_tail", "all_dummy", "ineligible"):
            results = []
            for enabled in (False, True):
                p["_MERGE_MTP_DP_SYNC"] = enabled
                r = runner(p, rank, ranks_per_group)
                main_count = 4 + 2 * rank
                global_max = 4 + 2 * (ranks_per_group - 1)
                role = 1 if scenario == "short_tail" else 3
                padded = rank % 3 == 2
                if scenario == "all_dummy" or rank % 3:
                    role = -1
                if scenario == "ineligible" and rank == 0:
                    role = 0
                r._mtp_dp_sync_role = role
                draft_count = (global_max if padded else main_count) if role == -1 else (role or 3)
                before = calls
                main = r._sync_metadata_across_dp(main_count, cudagraph_mode=Mode.NONE,
                                                  allow_dp_padding=padded)
                draft = r._sync_metadata_across_dp(draft_count, is_draft_model=True)
                expected_calls = 1 if enabled and scenario != "ineligible" else 2
                assert calls - before == expected_calls, (scenario, rank, calls - before)
                assert r.__dict__.get("_merged_mtp_dp_sync") is None
                results.append((main[0], main[1].tolist(), main[2],
                                draft[0], draft[1].tolist(), draft[2]))
                # A further draft call must participate in the original sync.
                before = calls
                r._sync_metadata_across_dp(draft_count, is_draft_model=True)
                assert calls == before + 1
            assert results[0] == results[1], (scenario, global_rank, results)
        # A prediction mismatch must fail without any process issuing another
        # metadata collective. All ranks deliberately exercise the same error.
        p["_MERGE_MTP_DP_SYNC"] = True
        r = runner(p, rank, ranks_per_group)
        r._mtp_dp_sync_role = rank + 1
        r._sync_metadata_across_dp(rank + 1)
        before = calls
        try:
            r._sync_metadata_across_dp(rank + 2, is_draft_model=True)
        except RuntimeError as exc:
            assert "prediction" in str(exc)
        else:
            raise AssertionError("incorrect prediction was accepted")
        assert calls == before
        assert r.__dict__.get("_merged_mtp_dp_sync") is None
        verify_gloo_overlap(torch, dist, group, rank, ranks_per_group)
        dist.barrier(group=group)
        if global_rank == 0:
            print(f"PASS: real Gloo, {groups} groups x {ranks_per_group} ranks; "
                  "original/merged outputs and mixed async/sync protocol verified", flush=True)
        if options and options.benchmark:
            record = gloo_benchmark(torch, dist, group, global_rank, rank, own_members, options, counted_reduce)
            records = [None] * (ranks_per_group * groups)
            dist.all_gather_object(records, record)
            if global_rank == 0:
                write_benchmark(records, memberships, options, torch)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--gloo" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gloo", action="store_true")
        parser.add_argument("--torchrun", action="store_true", help="use RANK/WORLD_SIZE and env:// rendezvous")
        parser.add_argument("--ranks-per-group", type=int)
        parser.add_argument("--groups", type=int, default=2)
        parser.add_argument("--group-layout", choices=("contiguous", "interleaved"), default="contiguous")
        parser.add_argument("--benchmark", action="store_true", help="run untimed correctness before full-call timing")
        parser.add_argument("--scenario", choices=("all_active", "mixed", "short_tail", "all_dummy", "ineligible"),
                            default="mixed")
        parser.add_argument("--requests-per-rank", type=int, default=1)
        parser.add_argument("--warmup", type=int, default=20)
        parser.add_argument("--blocks", type=int, default=20, help="alternate ABBA and BAAB blocks")
        parser.add_argument("--samples-per-arm", type=int, default=10)
        parser.add_argument("--output", help="JSON containing every rank sample and per-sample group maxima")
        args = parser.parse_args()
        if args.groups < 1:
            parser.error("need at least one group")
        if args.torchrun:
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                parser.error("--torchrun requires RANK and WORLD_SIZE from torchrun")
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size % args.groups:
                parser.error("WORLD_SIZE must be divisible by --groups")
            ranks_per_group = world_size // args.groups
            if args.ranks_per_group is not None and args.ranks_per_group != ranks_per_group:
                parser.error("--ranks-per-group does not match WORLD_SIZE / --groups")
            args.ranks_per_group = ranks_per_group
        elif args.ranks_per_group is None:
            args.ranks_per_group = 3
        if args.ranks_per_group < 2 or args.groups < 1:
            parser.error("need at least two ranks per group and one group")
        if args.requests_per_rank < 1 or args.warmup < 0 or args.blocks < 2 or args.samples_per_arm < 1:
            parser.error("need positive requests/samples, nonnegative warmup, and at least two blocks (ABBA + BAAB)")
        if args.benchmark and not args.output:
            parser.error("--benchmark requires --output to retain raw samples")
        if args.output:
            args.output = str(Path(args.output).resolve())
        if args.torchrun:
            gloo_worker(int(os.environ["RANK"]), args.ranks_per_group, args.groups, "env://", args)
        else:
            import torch.multiprocessing as mp

            with tempfile.TemporaryDirectory(prefix="mtp-dp-gloo-") as tmp:
                mp.spawn(gloo_worker, args=(args.ranks_per_group, args.groups, (Path(tmp) / "store").as_uri(), args),
                         nprocs=args.ranks_per_group * args.groups, join=True)
    else:
        unittest.main(verbosity=2)
