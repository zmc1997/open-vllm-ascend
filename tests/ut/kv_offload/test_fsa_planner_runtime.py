# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the production planner dispatch and allocation methods.

Array storage and NPU submission are emulated. These checks do not execute a
device kernel, capture/replay an ACL graph, or validate hardware performance.
"""

import ast
import contextlib
import ctypes
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(os.environ.get("VLLM_ASCEND_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
SOURCE = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py"


class Tensor(np.ndarray):
    device = NS(type="cpu", index=0)

    def size(self, axis):
        return self.shape[axis]

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return self.dtype.itemsize

    def data_ptr(self):
        return self.__array_interface__["data"][0]

    def is_contiguous(self):
        return self.flags.c_contiguous

    def stride(self, axis):
        return self.strides[axis] // self.itemsize

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], np.dtype):
            return np.ndarray.view(self, shape[0])
        return self.reshape(*shape)

    def repeat(self, *repeats):
        return np.tile(np.asarray(self), repeats).view(Tensor)

    def pin_memory(self):
        return self

    def copy_(self, value, **_):
        self[...] = value
        return self

    def fill_(self, value):
        self.fill(value)
        return self

    def record_stream(self, _stream):
        pass


def factory(function):
    def allocate(*args, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("pin_memory", None)
        return function(*args, **kwargs).view(Tensor)

    return allocate


def load_manager():
    torch = NS(
        Tensor=Tensor,
        Size=tuple,
        bfloat16=np.dtype("uint16"),
        int8=np.dtype("int8"),
        int16=np.dtype("int16"),
        int32=np.dtype("int32"),
        int64=np.dtype("int64"),
        iinfo=np.iinfo,
        tensor=factory(np.array),
        **{name: factory(getattr(np, name)) for name in ("empty", "zeros", "full", "arange")},
    )
    npu = NS(Stream=Mock(), synchronize=Mock())
    namespace = dict(
        torch=torch,
        torch_npu=NS(npu=npu),
        ctypes=ctypes,
        os=os,
        cdiv=lambda n, d: (n + d - 1) // d,
        logger=Mock(),
        envs=NS(
            VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY=str(Path("planner.so").resolve()),
            VLLM_ASCEND_FSA_ASYNC_PLAN=True,
            VLLM_ASCEND_FSA_REUSE_WRITEBACK_LAYOUT=True,
        ),
        is_forward_context_available=lambda: False,
    )
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseKVOffloadManager")
    names = {
        "_as_cache_tuple",
        "_register_offload_layers",
        "register_kv_caches",
        "_initialize_cpu_lru_planner",
        "_preflight_fsa_plan_ipc",
        "_load_fused_device_planner",
        "_init_fused_device_planner",
        "_get_fused_device_planner_layer_state",
        "allocate_fused_overlap_membership_map",
        "prepare_fused_overlap_external_plan",
        "_validate_fused_overlap_external_plan_inputs",
        "_join_fsa_plan_ipc",
        "join_fsa_plan_ipc_if_needed",
        "close_fsa_plan_ipc",
    }
    constants = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith(("FSA_", "OFFLOAD_")) for t in n.targets)
    ]
    cls = ast.ClassDef(
        name=original.name,
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *constants,
                *(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fsa_plan_payload_words"),
                cls,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    manager = namespace["SparseKVOffloadManager"]()
    manager.tp_rank, manager.tp_size = 0, 2
    manager.topk, manager.topk_buffer_size = 7, 128
    manager.max_num_topk_rows, manager.max_num_tokens, manager.max_model_len = 3, 3, 2 << 20
    manager.use_fused_overlap, manager.fused_async_plan = True, False
    manager._fused_device_plan_library = None
    manager.fused_overlap_membership_map = None
    manager.fused_overlap_membership_map_rows = 0
    manager.fused_overlap_plan_owner_layer_id = None
    manager.fused_overlap_plan_topk = None
    manager.fused_overlap_plan_num_tokens = 0
    manager.fused_overlap_plan_membership_map = None
    manager._get_offload_layer_id = int
    manager.tp_group = NS(broadcast=Mock(), barrier=Mock())
    return manager, namespace


def library_environment(library, cores=48, ub=192 * 1024):
    utilities = NS(AscendDeviceType=NS(A3="a3"), get_ascend_device_type=lambda: "a3")
    resources = NS(get_vector_resources=Mock(return_value=(cores, ub)))
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(ctypes, "CDLL", return_value=library))
    stack.enter_context(
        patch.dict(
            "sys.modules",
            {
                "vllm_ascend.utils": utilities,
                "vllm_ascend.ops.triton.kernel_resources": resources,
            },
        )
    )
    return stack


class TestFSAPlannerRuntime(unittest.TestCase):
    def test_failed_ipc_rejects_new_work_before_accessing_inputs(self):
        manager, _ = load_manager()
        ipc = manager._fsa_plan_ipc = Mock()
        ipc.ensure_usable.side_effect = RuntimeError("transport failed")
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            manager.prepare_fused_overlap_external_plan("0", 1, None, None, None, None, None)
        ipc.ensure_usable.assert_called_once()
        manager.tp_group.broadcast.assert_not_called()

    def test_ipc_close_keeps_resources_until_drain_and_graph_retirement(self):
        for fail_stage in (None, "drain", "clear_graph_params", "destroy"):
            with self.subTest(fail_stage=fail_stage):
                manager, ns = load_manager()
                actions = []

                def action(name):
                    def run(*_):
                        actions.append(name)
                        if name == fail_stage:
                            raise RuntimeError(name)
                    return run

                ipc = manager._fsa_plan_ipc = NS(mark_failed=action("stop"), close=action("destroy"))
                manager.fused_overlap_membership_device = "npu:0"
                ns["torch_npu"].npu.synchronize = action("drain")
                cleanup = NS(
                    clear_all_attention_workspaces=action("clear_workspaces"),
                    reset_all_graph_params=action("clear_graph_params"),
                    reset_model_runner_graph_manager=action("clear_graphs"),
                )
                with patch.dict("sys.modules", {
                    "vllm_ascend.device_allocator.sleep_mem_optimized": NS(AclGraphSleepWakeupManager=cleanup),
                }):
                    if fail_stage:
                        with self.assertRaisesRegex(RuntimeError, fail_stage):
                            manager.close_fsa_plan_ipc(object())
                    else:
                        manager.close_fsa_plan_ipc(object())
                expected = ["stop", "drain", "clear_workspaces", "clear_graph_params", "clear_graphs", "destroy"]
                if fail_stage:
                    expected = expected[:expected.index(fail_stage) + 1]
                self.assertEqual(actions, expected)
                self.assertIs(manager._fsa_plan_ipc, ipc)

    def test_ipc_eager_ack_join_is_device_dependency_not_host_sync(self):
        manager, ns = load_manager()
        manager._fsa_plan_ipc = object()
        manager.fused_overlap_plan_owner_layer_id = 0
        manager.fused_plan_stream = object()
        current = Mock()
        ns["torch_npu"].npu.current_stream = Mock(return_value=current)
        ns["get_forward_context"] = Mock(return_value=NS(_fsa_plan_ipc_join=None))
        manager.join_fsa_plan_ipc_if_needed("0", capturing=False)
        current.wait_stream.assert_called_once_with(manager.fused_plan_stream)
        ns["torch_npu"].npu.synchronize.assert_not_called()
        current.synchronize.assert_not_called()
        current.reset_mock()
        manager.join_fsa_plan_ipc_if_needed("1", capturing=False)
        current.wait_stream.assert_not_called()
        manager.join_fsa_plan_ipc_if_needed("0", capturing=True)
        current.wait_stream.assert_not_called()
        ns["get_forward_context"].return_value = NS()
        manager.join_fsa_plan_ipc_if_needed("0", capturing=True)
        current.wait_stream.assert_called_once_with(manager.fused_plan_stream)

    def test_ipc_preflight_consensus_rejects_partial_admission(self):
        cases = (
            (("off", "off"), None),
            (("on", "off"), "TP1: IPC disabled"),
            (("bad-library", "on"), "TP0: OSError: invalid ELF"),
            (("on", "missing-library"), "TP1: ValueError: FSA planner IPC requires"),
            (("on", "other-host"), "host-local TP group"),
            (("on", "different-geometry"), "wire geometry"),
            (("on", "hidden-boot-id"), "admitted"),
            (("on", "on"), "admitted"),
        )
        for modes, verdict in cases:
            for rank in (0, 1):
                with self.subTest(modes=modes, rank=rank):
                    manager, ns = load_manager()
                    ns["contextlib"] = contextlib
                    env, torch = ns["envs"], ns["torch"]
                    mode = modes[rank]
                    env.VLLM_ASCEND_FSA_PLAN_IPC = mode != "off"
                    if mode == "missing-library":
                        env.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY = None
                    expected_library = object()
                    loader = Mock(return_value=expected_library)
                    if mode == "bad-library":
                        loader.side_effect = OSError("invalid ELF")
                    if mode == "different-geometry":
                        manager.topk += 1
                    manager._load_fused_device_planner = loader
                    manager.tp_group.cpu_group = object()
                    ns["torch_npu"].npu.current_device = Mock(return_value=rank)
                    torch.device = Mock(return_value=NS(index=rank))
                    ns["open"] = Mock()
                    ns["open"].return_value.__enter__ = Mock(
                        return_value=NS(read=lambda: "b" if mode == "other-host" else "a")
                    )
                    ns["open"].return_value.__exit__ = Mock(return_value=False)
                    if mode == "hidden-boot-id":
                        ns["open"].side_effect = PermissionError("boot ID is hidden")
                    geometry = (7, 128, 3, 2 << 20)
                    packets = {
                        "off": (False, None, None, None),
                        "on": (True, "a", geometry, None),
                        "bad-library": (True, None, None, "OSError: invalid ELF"),
                        "missing-library": (
                            True, None, None,
                            "ValueError: FSA planner IPC requires VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY",
                        ),
                        "other-host": (True, "b", geometry, None),
                        "different-geometry": (True, "a", (8, *geometry[1:]), None),
                        "hidden-boot-id": (True, None, geometry, None),
                    }

                    def exchange(reports, packet, group):
                        self.assertIs(group, manager.tp_group.cpu_group)
                        self.assertEqual(packet, packets[mode])
                        reports[:] = [packets[name] for name in modes]

                    torch.distributed = NS(all_gather_object=Mock(side_effect=exchange))
                    if verdict == "admitted":
                        self.assertIs(manager._preflight_fsa_plan_ipc(), expected_library)
                    elif verdict is None:
                        self.assertIsNone(manager._preflight_fsa_plan_ipc())
                    else:
                        with self.assertRaisesRegex(RuntimeError, verdict):
                            manager._preflight_fsa_plan_ipc()
                    torch.distributed.all_gather_object.assert_called_once()
                    self.assertEqual(loader.call_count, int(mode not in ("off", "missing-library")))

    def test_actual_dimensions_and_resources_are_forwarded_without_deployment_gates(self):
        cases = (
            (1, 128, 1, 4096, 1, 24, 64 * 1024),
            (129, 384, 33, 1 << 21, 4, 40, 128 * 1024),
            (2048, 8192, 257, (1 << 24) + 1, 8, 48, 192 * 1024),
            (1, 128, 32768, 1 << 20, 4, 48, 192 * 1024),
        )
        for topk, capacity, rows, length, tp, cores, ub in cases:
            with self.subTest(topk=topk, capacity=capacity, rows=rows, length=length, tp=tp):
                manager, ns = load_manager()
                manager.topk, manager.topk_buffer_size = topk, capacity
                manager.max_num_topk_rows, manager.max_model_len, manager.tp_size = rows, length, tp
                library = Mock()
                with library_environment(library, cores, ub):
                    manager._init_fused_device_planner(NS(index=0))
                self.assertIs(manager._fused_device_plan_library, library)
                self.assertEqual(manager.fused_device_planner_vector_cores, cores)
                self.assertEqual(manager.fused_device_planner_ub_bytes, ub)
                self.assertEqual(manager.fused_device_planner_page_rows % 8, 0)
                self.assertLessEqual(manager.fused_device_planner_page_rows * 36, ub)
                wire_words = ((topk + 16) * 2 + 31) // 32 * 8 + 8
                self.assertEqual(manager.fused_device_planner_payload.shape, (rows, wire_words))
                self.assertEqual(manager.fused_device_planner_mapping.shape, (4 * rows,))
                self.assertEqual(manager.fused_device_planner_input_staging[1].shape, (rows, topk))
                self.assertEqual(len(library.fsa_lru_plan_async_v5.argtypes), 18)
                self.assertEqual(len(library.fsa_lru_publish_async_v4.argtypes), 7)

    def test_preserves_original_consumer_and_wire_constraints(self):
        for field, value in (
            ("topk", 0),
            ("topk", 2049),
            ("topk_buffer_size", 32768),
            ("max_num_topk_rows", 0),
            ("max_num_topk_rows", 32769),
            ("max_model_len", 0),
            ("max_model_len", 1 << 31),
        ):
            with self.subTest(field=field, value=value):
                manager, _ = load_manager()
                setattr(manager, field, value)
                with library_environment(Mock()), self.assertRaises(ValueError):
                    manager._init_fused_device_planner(NS(index=0))

    def test_stale_library_is_rejected_and_state_is_ordered_double_bank(self):
        manager, _ = load_manager()
        with library_environment(NS()), self.assertRaisesRegex(RuntimeError, "plan v5 / publish v4"):
            manager._init_fused_device_planner(NS(index=0))
        with library_environment(Mock()):
            manager._init_fused_device_planner(NS(index=0))
        state = manager._get_fused_device_planner_layer_state(0, False)
        self.assertTrue(np.all(state[0] == -1))
        self.assertTrue(np.all(state[1] == 0))
        self.assertEqual(state[2].shape, (3, 2, 2, 128))
        self.assertTrue(np.all(state[2][:, :, 0, :] == -1))
        self.assertTrue(np.all(state[2][:, :, 1, :] == np.arange(128)))
        self.assertIs(manager._get_fused_device_planner_layer_state(0, True), state)
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            manager._get_fused_device_planner_layer_state(1, True)

    def test_device_submission_respects_current_stream_and_refreshes_inputs(self):
        for async_plan, capturing, rank, tp in (
            (False, False, 0, 1),
            (False, False, 0, 4),
            (True, False, 0, 2),
            (False, True, 0, 4),
            (False, False, 1, 4),
            (True, True, 1, 4),
        ):
            with self.subTest(async_plan=async_plan, capturing=capturing, rank=rank, tp=tp):
                manager, ns = load_manager()
                torch, npu = ns["torch"], ns["torch_npu"].npu
                manager.tp_rank, manager.tp_size, manager.fused_async_plan = rank, tp, async_plan
                main, aux, save = Mock(npu_stream=101), Mock(npu_stream=202), Mock()
                npu.current_stream = Mock(return_value=main)
                npu.stream = Mock(side_effect=lambda _s: contextlib.nullcontext())
                manager.fused_plan_stream, manager.current_kv_save_stream = aux, save
                manager.sparse_kv_offload_cpp = Mock()
                library = Mock()
                library.fsa_lru_plan_async_v5.return_value = 0
                library.fsa_lru_publish_async_v4.return_value = 0
                with library_environment(library):
                    manager._init_fused_device_planner(NS(index=0))
                membership = manager.allocate_fused_overlap_membership_map(3)
                manager.fused_plan_metadata_npu = torch.zeros(4, dtype=torch.int32)
                if rank == 0:
                    manager._get_fused_device_planner_layer_state(0, False)
                req = torch.arange(2, dtype=torch.int64)
                topk = torch.zeros((2, 7), dtype=torch.int32)
                stable = torch.zeros(2, dtype=torch.int32)
                visible = torch.full(2, 1024, dtype=torch.int32)
                consumed = []
                library.fsa_lru_plan_async_v5.side_effect = (
                    lambda *_args, consumed=consumed, topk=topk: consumed.append(int(topk[0, 0])) or 0
                )
                for token in (37, 51):
                    topk[0, 0] = token
                    manager.prepare_fused_overlap_external_plan(
                        "0",
                        2,
                        topk,
                        req,
                        stable,
                        visible,
                        membership,
                        capturing=capturing,
                    )
                target_stream = aux if (async_plan or capturing) else main
                self.assertEqual(library.fsa_lru_plan_async_v5.call_count, 2 if rank == 0 else 0)
                self.assertEqual(consumed, [37, 51] if rank == 0 else [])
                self.assertEqual(library.fsa_lru_publish_async_v4.call_count, 2)
                self.assertEqual(library.fsa_lru_publish_async_v4.call_args.args[-1], target_stream.npu_stream)
                if rank == 0:
                    args = library.fsa_lru_plan_async_v5.call_args.args
                    self.assertEqual(args[9:], (2, 3, 48, 8, 7, 128, 192 * 1024, 2 << 20, target_stream.npu_stream))
                self.assertEqual(main.record_event.call_count, 2 if (async_plan or capturing) else 0)
                self.assertEqual(aux.wait_event.call_count, 2 if (async_plan or capturing) else 0)
                self.assertEqual(manager.tp_group.broadcast.call_count, 2)
                self.assertEqual(aux.wait_stream.call_count, 2 if capturing and rank == 0 and tp > 1 else 0)
                manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows.assert_not_called()
                manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows.assert_not_called()

    def test_registration_retires_old_graphs_before_replacing_buffers(self):
        manager, ns = load_manager()
        torch, npu = ns["torch"], ns["torch_npu"].npu
        manager.tp_rank, manager.tp_size = 0, 1
        manager.num_target_layers, manager.block_size = 1, 128
        manager.max_model_len = 1024
        manager.use_fused_copy_sfa = False
        manager._warmup_external_lru_planner_threads = Mock()
        manager._initialize_fused_writeback_descriptors = Mock()
        manager._initialize_paired_current_scatter = Mock()
        library = Mock()
        events = []
        npu.synchronize.side_effect = lambda *_: events.append("sync")
        lifecycle = NS(
            clear_all_attention_workspaces=lambda: events.append("clear_workspaces"),
            reset_all_graph_params=lambda: events.append("clear_graphs"),
        )
        allocation = manager._init_fused_device_planner

        def allocate(device):
            events.append("allocate")
            allocation(device)

        manager._init_fused_device_planner = allocate
        with (
            library_environment(library),
            patch.dict(
                "sys.modules",
                {
                    "vllm_ascend.device_allocator.sleep_mem_optimized": NS(AclGraphSleepWakeupManager=lifecycle),
                },
            ),
        ):
            previous = None
            for blocks in (3, 5):
                manager.kv_cache_config = NS(num_blocks=blocks)
                cache = (
                    torch.empty(0),
                    torch.empty(0),
                    torch.zeros((blocks, 128, 1, 4), dtype=torch.bfloat16),
                    torch.zeros((blocks, 128, 1, 2), dtype=torch.bfloat16),
                    torch.zeros((3, 128, 1, 4), dtype=torch.bfloat16),
                    torch.zeros((3, 128, 1, 2), dtype=torch.bfloat16),
                )
                manager.register_kv_caches({"layer.0": cache})
                self.assertTrue(manager._kv_caches_registered)
                self.assertIsNone(manager.fused_overlap_plan_owner_layer_id)
                self.assertIsNone(manager.fused_overlap_plan_membership_map)
                self.assertEqual(manager.fused_device_planner_layer_states, {})
                self.assertEqual(manager.addr_k_bases, [cache[4].data_ptr()])
                self.assertEqual(manager.gvas_k_bases, [cache[2].data_ptr()])
                self.assertFalse(hasattr(manager, "lru_token_mark_workspace"))
                self.assertFalse(hasattr(manager, "lru_token_pos_workspace"))
                manager._warmup_external_lru_planner_threads.assert_not_called()
                state = manager._get_fused_device_planner_layer_state(0, False)
                membership = manager.allocate_fused_overlap_membership_map(3)
                if previous is not None:
                    self.assertIsNot(manager.fused_device_planner_payload, previous[0])
                    self.assertIsNot(state, previous[1])
                    self.assertIsNot(membership, previous[2])
                previous = (manager.fused_device_planner_payload, state, membership)
                manager.fused_overlap_plan_owner_layer_id = 0
                manager.fused_overlap_plan_membership_map = membership
                state[0].fill_(99)
        self.assertEqual(events, ["allocate", "sync", "clear_workspaces", "clear_graphs", "allocate"])

    def test_original_cpu_registration_still_allocates_cpu_planning_state(self):
        manager, ns = load_manager()
        torch = ns["torch"]
        ns["envs"].VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY = None
        manager.tp_rank, manager.tp_size = 0, 1
        manager.num_target_layers, manager.block_size = 1, 128
        manager.max_model_len, manager.kv_cache_config = 1024, NS(num_blocks=3)
        manager.use_fused_copy_sfa = False
        manager._warmup_external_lru_planner_threads = Mock()
        manager._initialize_fused_writeback_descriptors = Mock()
        manager._initialize_paired_current_scatter = Mock()
        cache = (
            torch.empty(0),
            torch.empty(0),
            torch.zeros((3, 128, 1, 4), dtype=torch.bfloat16),
            torch.zeros((3, 128, 1, 2), dtype=torch.bfloat16),
            torch.zeros((3, 128, 1, 4), dtype=torch.bfloat16),
            torch.zeros((3, 128, 1, 2), dtype=torch.bfloat16),
        )
        manager.register_kv_caches({"layer.0": cache})
        self.assertIsNone(manager._fused_device_plan_library)
        self.assertEqual(manager.lru_token_mark_workspace.shape, (8, 1024))
        self.assertEqual(manager.lru_token_pos_workspace.shape, (8, 1024))
        self.assertEqual(manager.lru_slots_cpu_list[0].shape, (3, 128))
        self.assertTrue(np.all(manager.lru_slot_to_token_cpu_list[0] == -1))
        manager._warmup_external_lru_planner_threads.assert_called_once()

    def test_selection_state_follows_registration_without_resetting_live_cache(self):
        manager, ns = load_manager()
        torch = ns["torch"]
        with library_environment(Mock()):
            manager._init_fused_device_planner(NS(index=0))
        context = NS(capturing=False)
        source = ROOT / "vllm_ascend/attention/sfa_kv_offload.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAKVOffloadImpl")
        method = next(n for n in original.body if getattr(n, "name", None) == "_ensure_fused_overlap_selection_state")
        stride = next(n for n in tree.body if getattr(n, "name", None) == "_fsa_selection_status_stride")
        scope = dict(
            torch=torch,
            _FSA_SELECTION_STATUS_ALIGNMENT=8,
            get_forward_context=lambda: context,
            get_sparse_kv_offload_manager=lambda: manager,
        )
        module = ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                    stride,
                    method,
                ],
                type_ignores=[],
            )
        )
        exec(compile(module, str(source), "exec"), scope)
        ensure = scope["_ensure_fused_overlap_selection_state"]
        layers = [
            NS(
                max_num_topk_rows=3,
                sfa_sparse_topk=7,
                selection_kv_block_table=None,
                selection_kv_block_status=None,
                selection_membership_map=None,
                fused_overlap_last_req_ids=None,
                _fused_overlap_selection_capacity=None,
            )
            for _ in range(2)
        ]
        kwargs = dict(
            token_count=2,
            topk_head_count=1,
            topk=7,
            cache_blocks_per_row=1,
            device=Tensor.device,
            membership_device=Tensor.device,
        )
        saved = []
        for layer in layers:
            ensure(layer, **kwargs)
            layer.selection_kv_block_status.fill_(41)
            layer.fused_overlap_last_req_ids.fill_(91)
            saved.append(
                (
                    layer.selection_membership_map,
                    layer.selection_kv_block_status,
                    layer.fused_overlap_last_req_ids,
                )
            )
            # An ordinary new step, including graph capture, retains live state.
            context.capturing = True
            ensure(layer, **kwargs)
            self.assertIs(layer.selection_membership_map, saved[-1][0])
            self.assertIs(layer.selection_kv_block_status, saved[-1][1])
            self.assertTrue(np.all(layer.selection_kv_block_status == 41))
            self.assertTrue(np.all(layer.fused_overlap_last_req_ids == 91))
            context.capturing = False
        self.assertIs(saved[0][0], saved[1][0])
        # register_kv_caches retires this allocation before recreating the same
        # cache geometry. Each attention layer must notice the new ownership.
        manager.fused_overlap_membership_map = None
        manager.fused_overlap_membership_map_rows = 0
        context.capturing = True
        with self.assertRaisesRegex(RuntimeError, "preallocated before"):
            ensure(layers[0], **kwargs)
        context.capturing = False
        for layer, previous in zip(layers, saved):
            ensure(layer, **kwargs)
            self.assertIs(layer.selection_membership_map, manager.fused_overlap_membership_map)
            self.assertIsNot(layer.selection_membership_map, previous[0])
            self.assertIsNot(layer.selection_kv_block_status, previous[1])
            self.assertIsNot(layer.fused_overlap_last_req_ids, previous[2])
            self.assertTrue(np.all(layer.selection_kv_block_status == -1))
            self.assertTrue(np.all(layer.fused_overlap_last_req_ids == -1))
        self.assertIs(layers[0].selection_membership_map, layers[1].selection_membership_map)


if __name__ == "__main__":
    unittest.main()
