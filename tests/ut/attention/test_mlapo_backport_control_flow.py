"""Run production MLAPO routing/publication logic without importing device packages.

These are control-flow checks, not tensor numerical or NPU execution tests.
"""

import _imp
import argparse
import ast
import builtins
import itertools
import sys
import unittest
from contextlib import nullcontext
from enum import Enum
from importlib._bootstrap import _ImportLockContext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[3]
_SFA = "vllm_ascend/attention/sfa_kv_offload.py"
_MANAGER = "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py"


def _load_method(path, name, namespace):
    """Compile the actual method, keeping annotations lazy and imports isolated."""
    filename = _ROOT / path
    tree = ast.parse(filename.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace[name]


class _PreprocessType(Enum):
    NATIVE = 0
    MLAPO = 1


class _StaticQuantization:
    pass


class _View:
    """Only the view operations used to route selection writes, without data."""

    shape = (8, 4)

    def reshape(self, *_shape):
        return self

    def view(self, *_shape):
        return self

    def __getitem__(self, _index):
        return self


class TestMlapoBackportControlFlow(unittest.TestCase):
    def test_memcache_import_restores_flags_and_lock_on_success_and_failure(self):
        original_import = builtins.__import__
        for error in (None, ImportError("missing SDK"), RuntimeError("native import failed")):
            with self.subTest(error=type(error).__name__):
                runtime = SimpleNamespace(getdlopenflags=lambda: 2, setdlopenflags=Mock())
                store = Mock()
                store.init.return_value = 0

                def construct_store(runtime=runtime, store=store):
                    self.assertFalse(_imp.lock_held())
                    self.assertEqual(runtime.setdlopenflags.call_args.args, (2,))
                    return store

                def sdk_import(name, *args, import_error=error, runtime=runtime, constructor=construct_store, **kwargs):
                    if name == "memcache_hybrid":
                        self.assertTrue(_imp.lock_held())
                        self.assertEqual(runtime.setdlopenflags.call_args.args, (2 | 8,))
                        if import_error is not None:
                            raise import_error
                        return SimpleNamespace(DistributedObjectStore=constructor)
                    return original_import(name, *args, **kwargs)

                setup = _load_method(
                    "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py",
                    "_setup_store",
                    {
                        "sys": runtime,
                        "os": SimpleNamespace(RTLD_DEEPBIND=8),
                        "_ImportLockContext": _ImportLockContext,
                        "time": SimpleNamespace(sleep=Mock()),
                        "MEMCACHE_THREAD_START_WAIT_S": object(),
                        "logger": Mock(),
                    },
                )
                backend = SimpleNamespace(local_rank=0, _init_bm=False)
                with patch.object(builtins, "__import__", side_effect=sdk_import):
                    if error is None:
                        self.assertIs(setup(backend), store)
                        store.init.assert_called_once_with(0, init_bm=False)
                    else:
                        with self.assertRaises(type(error)) as caught:
                            setup(backend)
                        propagated = caught.exception.__cause__ if isinstance(error, ImportError) else caught.exception
                        self.assertIs(propagated, error)
                self.assertEqual(runtime.setdlopenflags.call_args.args, (2,))
                self.assertFalse(_imp.lock_held())

    def test_official_store_endpoint_is_shared_by_tp_and_separated_by_dp(self):
        tree = ast.parse((_ROOT / _MANAGER).read_text(encoding="utf-8"))
        initialize = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        start = next(
            i
            for i, node in enumerate(initialize.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "OffloadConfig"
        )
        stop = next(i for i in range(start, len(initialize.body)) if isinstance(initialize.body[i], ast.Assert))
        block = ast.fix_missing_locations(ast.Module(body=initialize.body[start : stop + 1], type_ignores=[]))
        code = compile(block, str(_ROOT / _MANAGER), "exec")
        for dp, tp in itertools.product(range(16), range(2)):
            with self.subTest(dp=dp, tp=tp):
                offload = SimpleNamespace(OffloadConfig=SimpleNamespace, Scene=SimpleNamespace(SHARED=object()))
                offload.initialize = Mock(return_value=0)
                namespace = {
                    "offload": offload,
                    "torch_npu": SimpleNamespace(npu=SimpleNamespace(current_device=lambda tp=tp: tp)),
                    "self": SimpleNamespace(tp_rank=tp, tp_size=2),
                    "parallel_config": SimpleNamespace(data_parallel_index=dp),
                    "actual_pool_size_bytes": 1 << 30,
                    "OFFLOAD_STORE_PORT_BASE": 8500,
                }
                exec(code, namespace)
                config = offload.initialize.call_args.args[0]
                self.assertEqual(config.store_url, f"tcp://127.0.0.1:{8500 + dp}")
                self.assertEqual((config.device_id, config.rank_id, config.world_size), (tp, tp, 2))
                self.assertEqual(config.alloc_size, (1 << 30) if tp == 0 else 0)
                self.assertIs(config.scene, offload.Scene.SHARED)

    def test_routing_preserves_existing_weight_preparation(self):
        config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=False))
        resolve = _load_method(
            _SFA,
            "_resolve_preprocess_type",
            {
                "PreprocessType": _PreprocessType,
                "AscendW8A8LinearMethod": _StaticQuantization,
                "get_ascend_config": lambda: config,
                "logger": SimpleNamespace(warning_once=Mock()),
            },
        )
        for condition in ("direct", "copy", "keep", "disabled", "quant", "unsupported"):
            with self.subTest(condition=condition):
                implementation = SimpleNamespace(
                    fused_qkv_a_proj=_StaticQuantization() if condition != "quant" else object(),
                    _get_layer_quant_method=lambda layer: layer,
                    use_fused_overlap=condition != "copy",
                    enable_mlapo=condition != "disabled",
                    _try_enable_type=Mock(return_value=condition != "unsupported"),
                )
                config.sparse_kv_offload_config.keep_device_kv_cache = condition == "keep"
                expected = _PreprocessType.MLAPO if condition == "direct" else _PreprocessType.NATIVE
                self.assertIs(resolve(implementation, object()), expected)
                self.assertEqual(
                    implementation._try_enable_type.call_count,
                    int(condition in ("direct", "unsupported")),
                )

    def test_producer_completion_replaces_external_copy_and_clears_stale_state(self):
        runtime = SimpleNamespace(current_stream=Mock(), stream=lambda _stream: nullcontext())
        publish = _load_method(
            _MANAGER,
            "offload_new_kv",
            {"torch_npu": SimpleNamespace(npu=runtime), "get_forward_context": lambda: SimpleNamespace()},
        )
        for rank, capturing, producer in itertools.product((0, 1), (False, True), (False, True)):
            with self.subTest(rank=rank, capturing=capturing, producer=producer):
                compute = Mock()
                runtime.current_stream.return_value = compute
                manager = SimpleNamespace(
                    use_fused_overlap=True,
                    tp_rank=rank,
                    current_kv_by_layer={},
                    producer_writeback_layers={0},
                    _get_offload_layer_id=lambda _layer: 0,
                    current_kv_save_stream=Mock(),
                    _offload_new_kv_on_current_stream=Mock(),
                )
                key, rope = object(), object()
                publish(
                    manager,
                    "layer",
                    object(),
                    object(),
                    object(),
                    None,
                    None,
                    key,
                    rope,
                    capturing=capturing,
                    writeback_in_producer=producer,
                )
                self.assertEqual(manager.current_kv_by_layer[0], (key, rope))
                self.assertEqual(0 in manager.producer_writeback_layers, producer)
                self.assertEqual(manager._offload_new_kv_on_current_stream.call_count, int(not producer))
                self.assertEqual(compute.record_event.call_count, int(rank == 0 and capturing))
                self.assertEqual(manager.current_kv_save_stream.wait_event.call_count, int(rank == 0 and capturing))

    def test_scatter_and_peer_publication_preserve_native_and_producer_dependencies(self):
        runtime = SimpleNamespace(current_stream=Mock())
        device_ops = SimpleNamespace(npu=runtime, npu_scatter_nd_update_=Mock())
        inject = _load_method(_MANAGER, "inject_current_kv_into_selection", {"torch_npu": device_ops})
        for rank, capturing, producer, owner, paired in itertools.product(
            (0, 1), (False, True), (False, True), (False, True), (False, True)
        ):
            with self.subTest(rank=rank, capturing=capturing, producer=producer, owner=owner, paired=paired):
                events = []
                save, plan = object(), object()
                runtime.current_stream.return_value = SimpleNamespace(
                    wait_stream=lambda stream, events=events, save=save: events.append(
                        "save" if stream is save else "plan"
                    )
                )
                device_ops.npu_scatter_nd_update_.side_effect = lambda *_args, events=events: events.append("scatter")

                def scatter_pair(*_args, events=events):
                    events.append("pair")
                    return True

                manager = SimpleNamespace(
                    _get_offload_layer_id=lambda _layer: 0,
                    current_kv_by_layer={0: (_View(), _View())},
                    producer_writeback_layers={0} if producer else set(),
                    fused_async_plan=False,
                    fused_plan_stream=plan,
                    current_kv_save_stream=save,
                    fused_plan_current_linear_slots_npu=_View(),
                    tp_rank=rank,
                    tp_size=2,
                    fused_overlap_plan_owner_layer_id=0 if owner else 1,
                    tp_group=SimpleNamespace(
                        broadcast=lambda *_args, events=events, **_kwargs: events.append("broadcast")
                    ),
                    fused_plan_status_npu=object(),
                    fused_current_scatter=scatter_pair if paired else None,
                )
                inject(manager, "layer", 1, _View(), _View(), capturing=capturing)
                expected = (["plan"] if capturing else []) + (["pair"] if paired else ["scatter", "scatter"])
                if capturing and rank == 0 and not producer:
                    expected.append("save")
                if (capturing or producer) and not owner:
                    expected.append("broadcast")
                self.assertEqual(events, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=_ROOT)
    options, arguments = parser.parse_known_args()
    _ROOT = options.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *arguments])
