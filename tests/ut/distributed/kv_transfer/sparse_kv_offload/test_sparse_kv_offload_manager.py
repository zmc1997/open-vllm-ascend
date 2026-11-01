import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)


class _FakeKVCacheSpec:
    def __init__(
        self,
        *,
        page_size_bytes,
        max_blocks_per_request,
        store_on_host,
        block_size=128,
    ):
        self.page_size_bytes = page_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.store_on_host = store_on_host
        self.block_size = block_size

    def max_memory_usage_bytes(self, _vllm_config):
        return self.max_blocks_per_request * self.page_size_bytes


def _make_memory_plan_inputs(max_num_seqs=2):
    specs = {
        "host.0": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "host.1": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "device.0": _FakeKVCacheSpec(
            page_size_bytes=512,
            max_blocks_per_request=100,
            store_on_host=False,
        ),
    }
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    alignment_reserve = 2 * manager_module._CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return specs, vllm_config, alignment_reserve


class TestSparseKVOffloadMemoryPlanning(unittest.TestCase):
    def test_memory_plan_is_limited_by_active_workload(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.npu_limit_blocks, 1000)
        self.assertEqual(budget.dram_limit_blocks, 1000)
        self.assertEqual(budget.workload_limit_blocks, 201)
        self.assertEqual(budget.final_num_blocks, 201)
        self.assertEqual(budget.final_planner_bytes, 201 * (2048 + 512))
        self.assertEqual(budget.planned_host_bytes, 201 * 2048)
        self.assertEqual(budget.planned_device_bytes, 201 * 512)
        self.assertEqual(budget.limiting_factor, "workload")

    def test_memory_plan_is_limited_by_dram_capacity(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 150 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.dram_limit_blocks, 150)
        self.assertEqual(budget.final_num_blocks, 150)
        self.assertEqual(budget.limiting_factor, "dram")

    def test_warning_when_dram_budget_caps_npu_utilization(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with self.assertLogs(manager_module.logger, level="WARNING") as logs:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 500 * 2048,
                keep_device_kv_cache=False,
            )
        self.assertTrue(any("dram_size_per_dp_GB" in line for line in logs.output))

    def test_no_warning_when_dram_budget_not_below_npu(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with patch.object(manager_module.logger, "warning_once") as mock_warn:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 1000 * 2048,
                keep_device_kv_cache=False,
            )
        mock_warn.assert_not_called()

    def test_keep_device_cache_counts_full_npu_page(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        total_page_size = 2048 + 512

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=125 * total_page_size,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=True,
        )

        self.assertEqual(budget.npu_limit_blocks, 125)
        self.assertEqual(budget.final_num_blocks, 125)
        self.assertEqual(budget.planned_device_bytes, 125 * total_page_size)
        self.assertEqual(budget.limiting_factor, "npu")

    def test_non_positive_capacity_produces_zero_blocks(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        for available_device_memory, dram_limit, limiting_factor in (
            (-1, alignment_reserve + 1000 * 2048, "npu"),
            (1000 * 512, alignment_reserve - 1, "dram"),
        ):
            with self.subTest(limiting_factor=limiting_factor):
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=available_device_memory,
                    dram_limit_bytes=dram_limit,
                    keep_device_kv_cache=False,
                )
                self.assertEqual(budget.final_num_blocks, 0)
                self.assertEqual(budget.final_planner_bytes, 0)
                self.assertEqual(budget.limiting_factor, limiting_factor)

    def test_memory_plan_rejects_invalid_spec_layouts(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        invalid_cases = (
            ({"device.0": specs["device.0"]}, "at least one host"),
            ({"host.0": specs["host.0"]}, "at least one device"),
            (
                {
                    **specs,
                    "device.1": _FakeKVCacheSpec(
                        page_size_bytes=512,
                        max_blocks_per_request=100,
                        store_on_host=False,
                        block_size=64,
                    ),
                },
                "one shared block size",
            ),
        )

        for invalid_specs, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan_sparse_kv_offload_memory(
                    kv_cache_spec=invalid_specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                )

    def test_cpu_pool_size_includes_per_layer_alignment_reserve(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=spec,
                )
                for name, spec in specs.items()
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_supports_uniform_specs(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        uniform_specs = UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs=specs,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=uniform_specs,
                )
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_rejects_missing_host_specs(self):
        specs, _, _ = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["device.0"],
                    kv_cache_spec=specs["device.0"],
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "host-resident"):
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)

    def _make_manager_init_inputs(self, dram_size_per_dp_gb=1):
        spec = _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[SimpleNamespace(layer_names=["host.0"], kv_cache_spec=spec)],
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_layers=MagicMock(return_value=1),
                max_model_len=128,
            ),
            parallel_config=SimpleNamespace(data_parallel_index=3),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=1,
            ),
            speculative_config=None,
        )
        offload_config = SimpleNamespace(
            topk_buffer_size=1,
            topk=1,
            use_fused_overlap=False,
            dram_size_per_dp_GB=dram_size_per_dp_gb,
        )
        return vllm_config, kv_cache_config, offload_config

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size in ((0, planned_pool_size), (1, 0)):
            with self.subTest(rank=rank):
                offload_backend = SimpleNamespace(
                    OffloadConfig=lambda: SimpleNamespace(),
                    Scene=SimpleNamespace(SHARED="shared"),
                    initialize=MagicMock(return_value=0),
                )
                tp_group = SimpleNamespace(barrier=MagicMock())

                with (
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_rank",
                        return_value=rank,
                    ),
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_world_size",
                        return_value=2,
                    ),
                    patch.object(
                        manager_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        manager_module,
                        "get_sparse_kv_offload_cpu_pool_size_bytes",
                        return_value=planned_pool_size,
                    ),
                    patch.object(
                        manager_module,
                        "offload",
                        offload_backend,
                        create=True,
                    ),
                    patch.object(
                        manager_module.torch,
                        "zeros",
                        return_value=MagicMock(),
                    ),
                    patch.object(
                        manager_module.torch,
                        "empty",
                        return_value=MagicMock(),
                    ),
                    patch.object(SparseKVOffloadManager, "_build_cpp"),
                ):
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )

                initialized_config = offload_backend.initialize.call_args.args[0]
                self.assertEqual(
                    initialized_config.reserve_size,
                    planned_pool_size,
                )
                self.assertEqual(
                    initialized_config.alloc_size,
                    expected_alloc_size,
                )
                self.assertEqual(initialized_config.world_size, 2)
                self.assertEqual(initialized_config.rank_id, rank)
                self.assertEqual(initialized_config.scene, "shared")
                self.assertEqual(initialized_config.store_url, "tcp://127.0.0.1:8503")
                tp_group.barrier.assert_called_once_with()

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        offload_backend = SimpleNamespace(
            OffloadConfig=MagicMock(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(
                manager_module,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(
                manager_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                manager_module,
                "get_tp_group",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=(1 << 30) + 1,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(SparseKVOffloadManager, "_build_cpp"),
            self.assertRaisesRegex(ValueError, "exceeds DRAM limit"),
        ):
            SparseKVOffloadManager(
                vllm_config,
                kv_cache_config,
                offload_config,
            )

        offload_backend.OffloadConfig.assert_not_called()
        offload_backend.initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestHostKVRegistration(unittest.TestCase):
    @staticmethod
    def _make_fused_overlap_manager():
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.tp_rank = 0
        manager.num_layers = 1
        manager.offload_layer_names = ["layer.0"]
        manager.kv_cache_config = SimpleNamespace(num_blocks=3)
        manager.topk_buffer_size = 256
        manager.topk = 256
        manager.block_size = 128
        manager.max_num_tokens = 2
        manager.max_num_topk_rows = 2
        manager.use_fused_copy_sfa = False
        manager.use_fused_overlap = True
        manager._register_offload_layers = MagicMock()
        manager.max_model_len = 128
        manager._warmup_external_lru_planner_threads = MagicMock()
        manager._initialize_fused_writeback_descriptors = MagicMock()
        manager._initialize_paired_current_scatter = MagicMock()
        manager.tp_group = SimpleNamespace(broadcast=MagicMock())
        return manager

    @staticmethod
    def _make_fused_overlap_cache(num_blocks=3):
        torch = manager_module.torch
        return (
            torch.empty(0),
            torch.empty(0),
            torch.empty((num_blocks, 128, 1, 4), dtype=torch.bfloat16),
            torch.empty((num_blocks, 128, 1, 2), dtype=torch.bfloat16),
            torch.empty((2, 256, 1, 4), dtype=torch.bfloat16),
            torch.empty((2, 256, 1, 2), dtype=torch.bfloat16),
        )

    def test_fused_overlap_registers_replacement_cpu_pools(self):
        torch = manager_module.torch
        manager = self._make_fused_overlap_manager()
        originals = {name: getattr(torch, name) for name in ("zeros", "empty", "full", "arange", "tensor")}

        def cpu_factory(original):
            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                kwargs.pop("pin_memory", None)
                return original(*args, **kwargs)

            return allocate

        with (
            patch.object(torch, "zeros", side_effect=cpu_factory(originals["zeros"])),
            patch.object(torch, "empty", side_effect=cpu_factory(originals["empty"])),
            patch.object(torch, "full", side_effect=cpu_factory(originals["full"])),
            patch.object(torch, "arange", side_effect=cpu_factory(originals["arange"])),
            patch.object(torch, "tensor", side_effect=cpu_factory(originals["tensor"])),
            patch.object(torch.Tensor, "pin_memory", lambda value: value),
            patch.object(manager_module.torch_npu.npu, "Stream") as stream,
        ):
            for num_blocks in (3, 5):
                with self.subTest(num_blocks=num_blocks):
                    manager.kv_cache_config.num_blocks = num_blocks
                    cache = self._make_fused_overlap_cache(num_blocks)
                    manager.register_kv_caches({"layer.0": cache})
                    self.assertIs(manager.k_caches_cpu[0], cache[2])
                    self.assertIs(manager.v_caches_cpu[0], cache[3])
                    self.assertEqual(manager.gvas_k_bases, [cache[2].data_ptr()])
                    self.assertEqual(manager.gvas_v_bases, [cache[3].data_ptr()])
                    self.assertEqual(manager.k_caches_cpu[0].numel() // 4, num_blocks * 128)
                    self.assertEqual(manager.v_caches_cpu[0].numel() // 2, num_blocks * 128)
                    self.assertEqual(manager.cpu_block_lens, [(1024, 512)])

        self.assertEqual(stream.call_count, 4)
        self.assertEqual(manager._warmup_external_lru_planner_threads.call_count, 2)

    def test_fused_overlap_rejects_invalid_cpu_pools_before_startup(self):
        torch = manager_module.torch
        invalid_pools = {
            "non_cpu": lambda pool: torch.empty_like(pool, device="meta"),
            "dtype": lambda pool: pool.to(torch.float16),
            "noncontiguous": lambda pool: pool.transpose(0, 1),
            "partial_row": lambda pool: torch.empty(pool.numel() + 1, dtype=pool.dtype),
            "zero_capacity": lambda pool: pool[:0],
            "unequal_capacity": lambda pool: pool[:2],
            "missing": lambda pool: None,
        }
        for pool_index in (2, 3):
            for reason, invalidate in invalid_pools.items():
                with self.subTest(pool_index=pool_index, reason=reason):
                    manager = self._make_fused_overlap_manager()
                    cache = list(self._make_fused_overlap_cache())
                    cache[pool_index] = invalidate(cache[pool_index])
                    with (
                        patch.object(manager_module.torch_npu.npu, "Stream") as stream,
                        self.assertRaises(ValueError),
                    ):
                        manager.register_kv_caches({"layer.0": tuple(cache)})
                    stream.assert_not_called()
                    manager._warmup_external_lru_planner_threads.assert_not_called()
                    manager.tp_group.broadcast.assert_not_called()
