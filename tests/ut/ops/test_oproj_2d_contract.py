# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the production two-dimensional o_proj path.

Run directly (no vLLM, torch_npu, NPU initialization or pytest conftest):
  python test_oproj_2d_contract.py --source-root /path/to/vllm-ascend

AST loading executes the actual production methods. Only device kernels and
collectives are replaced. The reference is a full, unsliced X @ W with distinct
DP inputs, including an idle replica. This checks routing and compensation, not
CANN support, graph capture, BF16 rounding equivalence or device performance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import logging
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[3]


def load_nodes(relative, names, namespace):
    """Execute selected real definitions, retaining their method bodies."""
    source = SOURCE_ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = []
    for name in names:
        scope = tree.body
        for part in name.split("."):
            node = next(n for n in scope if getattr(n, "name", None) == part)
            scope = getattr(node, "body", [])
        selected.append(copy.deepcopy(node))
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class Collectives:
    """Blocking CPU collectives with rank-local ordering and bounded waits."""

    def __init__(self):
        self.condition = threading.Condition()
        self.slots = {}
        self.error = None
        self.log = []

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def run(self, group, kind, tensor):
        key = (group.members, group.sequence)
        group.sequence += 1
        with self.condition:
            slot = self.slots.setdefault(key, {"kind": kind, "inputs": {}})
            if slot["kind"] != kind:
                raise AssertionError(f"collective order mismatch: {key}: {slot['kind']} != {kind}")
            slot["inputs"][group.rank_in_group] = tensor.clone()
            if len(slot["inputs"]) == group.world_size:
                inputs = [slot["inputs"][i] for i in range(group.world_size)]
                self.log.append((group.members, kind, tuple(tensor.shape)))
                if kind == "a2a":
                    chunks = [x.chunk(group.world_size, dim=0) for x in inputs]
                    outputs = [torch.cat([chunks[s][d] for s in range(group.world_size)])
                               for d in range(group.world_size)]
                else:
                    total = torch.stack(inputs).sum(dim=0)
                    outputs = list(total.chunk(group.world_size, dim=0)) if kind == "rs" else [total] * group.world_size
                slot["outputs"] = outputs
                self.condition.notify_all()
            ready = self.condition.wait_for(lambda: "outputs" in slot or self.error is not None, timeout=10)
            if not ready:
                raise AssertionError(f"unmatched collective {key} ({kind})")
            if self.error is not None:
                raise RuntimeError("another rank failed") from self.error
            return slot["outputs"][group.rank_in_group].clone()


class Group:
    def __init__(self, world, members, rank):
        self.world, self.members = world, tuple(members)
        self.rank_in_group = self.members.index(rank)
        self.world_size = len(members)
        self.device_group = self
        self.sequence = 0

    def all_reduce(self, value):
        return self.world.run(self, "ar", value)

    def reduce_scatter(self, value, dim=0):
        assert dim == 0
        return self.world.run(self, "rs", value)

    def _get_backend(self, device):
        return self

    def get_hccl_comm_name(self, rank):
        assert rank == self.rank_in_group
        return self


class TorchCPU:
    """Permit production communicator setup without registering an NPU backend."""

    @staticmethod
    def device(value):
        return torch.device("cpu" if str(value).startswith("npu") else value)

    def __getattr__(self, name):
        return getattr(torch, name)


class OProjContract(unittest.TestCase):
    def setUp(self):
        self.local = threading.local()
        self.world = Collectives()
        self.finegrained = types.SimpleNamespace(
            oproj_tensor_parallel_size=4, oproj_communication="reduce_scatter", oproj_quant_pack=False,
            oproj_quant_pack_block_size=1024,
            oproj_native_communication=False,
        )
        self.config = types.SimpleNamespace(finegrained_tp_config=self.finegrained)
        self.namespace = {
            "torch": TorchCPU(),
            "Parameter": torch.nn.Parameter,
            "logger": logging.getLogger(__name__),
            "get_tp_group": lambda: self.local.tp,
            "get_otp_group": lambda: self.local.otp,
            "get_otp_total_group": lambda: self.local.total,
            "get_oproj_native_communicator": lambda: types.SimpleNamespace(
                all_to_all=lambda send, recv: recv.copy_(self.world.run(self.local.otp, "a2a", send)),
                all_reduce=lambda value: self.world.run(self.local.total, "ar", value),
            ),
            "get_tensor_model_parallel_rank": lambda: self.local.tp.rank_in_group,
            "get_tensor_model_parallel_world_size": lambda: self.local.tp.world_size,
            "get_ascend_config": lambda: self.config,
            "oproj_tp_enable": lambda: True,
            "mlp_tp_enable": lambda: False,
            "tensor_model_parallel_all_reduce": lambda x: self.local.tp.all_reduce(x),
            "split_tensor_along_last_dim": lambda x, num_partitions: x.chunk(num_partitions, dim=-1),
            "COMPRESSED_TENSORS_METHOD": "compressed-tensors",
            "maybe_trans_nz": lambda x: x,
        }

        def a2a(output, input_, group, **kwargs):
            output.copy_(self.world.run(group, "a2a", input_))

        def quantize(x, scale, reciprocal, offset):
            return torch.round(x.float() * reciprocal.float() + offset.float()).clamp(-128, 127).to(torch.int8)

        def quant_pack(x, reciprocal, offset, exchange_size, *, block_size):
            self.assertEqual(block_size, self.finegrained.oproj_quant_pack_block_size)
            self.assertEqual((reciprocal.dtype, offset.dtype), (torch.float32, torch.float32))
            self.assertEqual((reciprocal.data_ptr(), offset.data_ptr()), self.local.prepared_addresses)
            q = quantize(x, None, reciprocal, offset)
            return q.reshape(x.shape[0], exchange_size, -1).transpose(0, 1).contiguous()

        def prepare_parameters(reciprocal, offset):
            self.assertEqual((reciprocal.dtype, offset.dtype), (torch.bfloat16, torch.bfloat16))
            self.local.prepare_count += 1
            prepared = reciprocal.float(), offset.float()
            self.local.prepared_addresses = tuple(value.data_ptr() for value in prepared)
            return prepared

        def quant_mm(x, w, scale, bias=None, output_dtype=None):
            result = x.float() @ w.float()
            if bias is not None:
                result += bias.float()
            return (result * scale.float()).to(output_dtype)

        def mmrs(x, w, hcom, world_size, *, x2_scale=None, bias=None, output_dtype=None, **kwargs):
            self.assertEqual(world_size, hcom.world_size)
            partial = quant_mm(x, w, x2_scale, bias=bias, output_dtype=output_dtype)
            return hcom.reduce_scatter(partial)

        self.namespace["dist"] = types.SimpleNamespace(all_to_all_single=a2a)
        self.namespace["DeviceOperator"] = types.SimpleNamespace(npu_mm_reduce_scatter_base=mmrs)
        self.namespace["torch_npu"] = types.SimpleNamespace(npu_quant_matmul=quant_mm)
        self.quant_patch = patch.object(torch.ops.vllm, "quantize", quantize, create=True)
        self.quant_patch.start()
        self.addCleanup(self.quant_patch.stop)

        class RowLayer(torch.nn.Module):
            pass

        self.RowLayer = RowLayer
        self.namespace["RowParallelLinear"] = RowLayer
        load_nodes("vllm_ascend/quantization/method_adapters.py",
                   ["AscendLinearMethod.apply", "AscendLinearMethod.process_weights_after_loading"], self.namespace)
        self.Adapter = type("AscendLinearMethod", (), {
            "apply": self.namespace.pop("apply"),
            "process_weights_after_loading": self.namespace.pop("process_weights_after_loading"),
        })
        load_nodes("vllm_ascend/quantization/methods/w8a8_static.py",
                   ["AscendW8A8LinearMethod.apply", "AscendW8A8LinearMethod.process_weights_after_loading"], self.namespace)
        self.Scheme = type("AscendW8A8LinearMethod", (), {
            "apply": self.namespace.pop("apply"),
            "process_weights_after_loading": self.namespace.pop("process_weights_after_loading"),
        })
        self.namespace.update(AscendLinearMethod=self.Adapter, AscendW8A8LinearMethod=self.Scheme)
        # update_attrs imports these names locally; supply only their types,
        # never import the NPU package or execute its initialization.
        modules = {}
        for name in ("vllm_ascend", "vllm_ascend.quantization", "vllm_ascend.quantization.method_adapters",
                     "vllm_ascend.quantization.methods", "vllm_ascend.ops", "vllm_ascend.ops.triton",
                     "vllm_ascend.ops.triton.oproj_quant_pack"):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        modules["vllm_ascend.quantization.method_adapters"].AscendLinearMethod = self.Adapter
        modules["vllm_ascend.quantization.methods"].AscendW8A8LinearMethod = self.Scheme
        modules["vllm_ascend.ops.triton.oproj_quant_pack"].quantize_pack = quant_pack
        modules["vllm_ascend.ops.triton.oproj_quant_pack"].prepare_parameters = prepare_parameters
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        load_nodes("vllm_ascend/ops/linear_op.py",
                   ["CustomLinearOp", "CustomRowParallelOp", "OProjRowParallelOp"], self.namespace)
        load_nodes("vllm_ascend/attention/sfa_v1.py", ["AscendSFAImpl.forward"], self.namespace)
        self.sfa_forward = self.namespace.pop("forward")

    def check_route(self, tp_size, dp_size, mode, input_is_parallel=True, idle_sfa=False, repeat=1, quant_pack=False, native=False):
        self.world = Collectives()
        self.finegrained.oproj_tensor_parallel_size = tp_size * dp_size
        self.finegrained.oproj_communication = mode
        self.finegrained.oproj_quant_pack = quant_pack
        self.finegrained.oproj_native_communication = native
        n, k, h = 3, 32, 5
        generator = torch.Generator().manual_seed(704)
        inputs = torch.randint(-2, 3, (dp_size, n, k), generator=generator).to(torch.bfloat16)
        inputs[-1].zero_()  # The idle DP must still receive compensation correctly.
        full_weight = torch.randint(-1, 2, (k, h), generator=generator, dtype=torch.int8)
        full_weight[:, 0] = 1  # Guarantee a nonzero compensation term.
        scales = torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0])
        quant_bias = (-3 * full_weight.int().sum(dim=0)).to(torch.int32)
        outputs, errors, shard_ranks = {}, [], {}

        def worker(rank):
            try:
                self.local.prepare_count = 0
                d, t = divmod(rank, tp_size)
                self.local.tp = Group(self.world, range(d * tp_size, (d + 1) * tp_size), rank)
                self.local.otp = Group(self.world, range(t, tp_size * dp_size, tp_size), rank)
                self.local.total = Group(self.world, range(tp_size * dp_size), rank)
                layer = self.RowLayer()
                layer.prefix = "model.layers.0.self_attn.o_proj"
                layer.input_size, layer.output_size = k, h
                layer.input_is_parallel, layer.reduce_results = input_is_parallel, True
                layer.skip_bias_add, layer.return_bias, layer.bias = False, True, None
                layer.params_dtype = torch.bfloat16
                op = self.namespace["OProjRowParallelOp"](layer)
                layer.custom_op = op
                layer.tp_rank, layer.tp_size = op.tp_rank, op.tp_size
                shard_ranks[rank] = layer.tp_rank
                layer.input_size_per_partition = k // layer.tp_size
                # Loading by the rank selected by production code makes a rank
                # permutation bug observable against the full unsliced reference.
                shard = full_weight.chunk(layer.tp_size, dim=0)[layer.tp_rank]
                layer.weight = torch.nn.Parameter(shard.t().contiguous(), requires_grad=False)
                for name, value in {
                    "input_scale": torch.ones(1, dtype=torch.bfloat16),
                    "input_offset": torch.tensor([3], dtype=torch.int8),
                    "quant_bias": quant_bias.clone(), "deq_scale": scales.clone(),
                    "weight_scale": torch.ones(h, 1, dtype=torch.bfloat16),
                    "weight_offset": torch.zeros(h, 1, dtype=torch.bfloat16),
                }.items():
                    setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
                scheme = self.Scheme()
                adapter = self.Adapter()
                adapter.quant_method = scheme
                layer.quant_method = adapter
                op.update_attrs()
                adapter.process_weights_after_loading(layer)
                if quant_pack:
                    self.assertEqual(layer.aclnn_input_scale_reciprocal.dtype, torch.bfloat16)
                    self.assertEqual(layer.aclnn_input_offset.dtype, torch.bfloat16)
                local_input = inputs[d].chunk(tp_size, dim=-1)[t].contiguous() if input_is_parallel else inputs[d]
                for _ in range(repeat):
                    if idle_sfa and d == dp_size - 1:
                        def project(dummy):
                            outputs[rank] = op.apply(dummy)[0]
                            return outputs[rank], None

                        project.tp_size = layer.tp_size
                        attention = types.SimpleNamespace(local_num_heads=k // tp_size, v_head_dim=1, o_proj=project)
                        dummy_output = torch.empty(n, h, dtype=torch.bfloat16)
                        result = self.sfa_forward(attention, layer.prefix, local_input, (), None, output=dummy_output)
                        self.assertEqual(result.count_nonzero().item(), 0)
                    else:
                        outputs[rank] = op.apply(local_input)[0]
                self.assertEqual(self.local.prepare_count, int(quant_pack))
            except BaseException as error:
                errors.append((rank, error))
                self.world.abort(error)

        threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(tp_size * dp_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(t.is_alive() for t in threads), "collective threads did not terminate")
        if errors:
            raise AssertionError(f"rank {errors[0][0]} failed: {errors[0][1]}") from errors[0][1]
        for rank, result in outputs.items():
            d, t = divmod(rank, tp_size)
            self.assertEqual(shard_ranks[rank], t * dp_size + d)
            expected = (inputs[d].float() @ full_weight.float()) * scales
            torch.testing.assert_close(result.float(), expected, rtol=0, atol=0)
        exchanges = {members for members, kind, _ in self.world.log if kind == "a2a"}
        self.assertEqual(exchanges, {tuple(range(t, tp_size * dp_size, tp_size)) for t in range(tp_size)})
        self.assertEqual(sum(kind == "a2a" for _, kind, _ in self.world.log), tp_size * repeat)

    def test_tp2_dp2_reduce_scatter(self):
        self.check_route(2, 2, "reduce_scatter")

    def test_legacy_tp1_reduce_scatter(self):
        self.check_route(1, 4, "reduce_scatter")

    def test_tp2_dp4_reduce_scatter(self):
        self.check_route(2, 4, "reduce_scatter")

    def test_tp2_dp2_all_reduce(self):
        self.check_route(2, 2, "all_reduce")

    def test_quant_pack_int8_exchange_with_idle_mtp(self):
        for block_size in (1024, 2048, 4096):
            with self.subTest(block_size=block_size):
                self.finegrained.oproj_quant_pack_block_size = block_size
                self.check_route(2, 2, "all_reduce", idle_sfa=True, repeat=5, quant_pack=True)

    def test_tp2_dp4_all_reduce(self):
        self.check_route(2, 4, "all_reduce")

    def test_native_route_never_uses_process_group_collectives(self):
        def unexpected(*args, **kwargs):
            self.fail("Native o_proj must not materialize HCCL process-group collectives")

        self.namespace["dist"] = types.SimpleNamespace(all_to_all_single=unexpected)
        with patch.object(Group, "all_reduce", unexpected):
            self.check_route(2, 2, "all_reduce", idle_sfa=True, repeat=5, quant_pack=True, native=True)

    def test_unsharded_input_is_sliced_by_outer_tp_once(self):
        self.check_route(2, 2, "all_reduce", input_is_parallel=False)

    def test_mmrs_nonzero_compensation_on_every_dp_output(self):
        self.check_route(2, 2, "matmul_reduce_scatter")

    def test_five_eager_mtp_calls_with_idle_dp(self):
        self.check_route(2, 2, "matmul_reduce_scatter", idle_sfa=True, repeat=5)

    def test_disabled_oproj_dummy_keeps_original_zero_output(self):
        self.namespace["oproj_tp_enable"] = lambda: False
        def unexpected(_):
            self.fail("disabled o_proj must not join collectives")

        attention = types.SimpleNamespace(local_num_heads=16, v_head_dim=1, o_proj=unexpected)
        result = self.sfa_forward(attention, "dummy", torch.ones(3, 16), (), None, output=torch.ones(3, 5))
        self.assertEqual(result.count_nonzero().item(), 0)


class ConfigurationContract(unittest.TestCase):
    def config(self, tp=2, dp=16, model_type="glm_moe_dsa"):
        return types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp,
                pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
            model_config=types.SimpleNamespace(is_moe=True, enforce_eager=False,
                hf_config=types.SimpleNamespace(model_type=model_type),
                hf_text_config=types.SimpleNamespace(model_type=model_type),
                architectures=["GlmMoeDsaForCausalLM"]),
            kv_transfer_config=types.SimpleNamespace(is_kv_consumer=True, is_kv_producer=False),
            additional_config={}, speculative_config=None,
        )

    def finegrained(self, values, config):
        namespace = {"logger": logging.getLogger(__name__), "enable_sp": lambda *args: False}
        load_nodes("vllm_ascend/ascend_config.py", ["FinegrainedTPConfig"], namespace)
        return namespace["FinegrainedTPConfig"](values, config)

    def test_total_tp_divides_combined_world_not_dp_alone(self):
        result = self.finegrained({"oproj_tensor_parallel_size": 4}, self.config(dp=2))
        self.assertEqual(result.oproj_tensor_parallel_size, 4)
        self.assertEqual(result.oproj_dp_size, 2)

    def test_quant_pack_requires_supported_tp_and_boolean(self):
        result = self.finegrained({"oproj_tensor_parallel_size": 4, "oproj_quant_pack": True}, self.config())
        self.assertTrue(result.oproj_quant_pack)
        for values in ({"oproj_quant_pack": True}, {"oproj_tensor_parallel_size": 4, "oproj_quant_pack": "true"}):
            with self.assertRaises(ValueError):
                self.finegrained(values, self.config())

    def test_quant_pack_block_size(self):
        self.assertEqual(self.finegrained({}, self.config()).oproj_quant_pack_block_size, 1024)
        for block_size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            with self.subTest(block_size=block_size):
                result = self.finegrained({"oproj_quant_pack_block_size": block_size}, self.config())
                self.assertEqual(result.oproj_quant_pack_block_size, block_size)
        for block_size in (None, False, True, "1024", 1024.0, -32, 0, 16, 33, 3072, 8192):
            with self.subTest(block_size=block_size), self.assertRaises(ValueError):
                self.finegrained({"oproj_quant_pack_block_size": block_size}, self.config())

    def test_reject_invalid_sizes_and_architecture(self):
        for config, size in [(self.config(), 3), (self.config(dp=3), 4),
                             (self.config(model_type="deepseek_v4"), 4)]:
            with self.subTest(size=size, config=config):
                with self.assertRaises((AssertionError, ValueError)):
                    self.finegrained({"oproj_tensor_parallel_size": size}, config)

    def test_oproj_alone_forces_dp_metadata_sync(self):
        namespace = {"lmhead_tp_enable": lambda: False, "oproj_tp_enable": lambda: True,
                     "embedding_tp_enable": lambda: False}
        load_nodes("vllm_ascend/utils.py", ["should_skip_allreduce_across_dp_group"], namespace)
        for draft in (False, True):
            self.assertFalse(namespace["should_skip_allreduce_across_dp_group"](self.config(), draft))

    def test_reject_unsupported_parallel_layouts(self):
        for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
            config = self.config()
            setattr(config.parallel_config, name, 2)
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.finegrained({"oproj_tensor_parallel_size": 4}, config)
        config = self.config()
        config.additional_config["enable_dsa_cp"] = True
        with self.assertRaises(AssertionError):
            self.finegrained({"oproj_tensor_parallel_size": 4}, config)


class GroupTopologyContract(unittest.TestCase):
    def check_group_builder_and_destroy(self, native):
        created, destroyed = {}, []

        def create(groups, local_rank, backend, group_name):
            created[group_name] = groups
            return types.SimpleNamespace(destroy=lambda: destroyed.append(group_name))

        parallel = types.SimpleNamespace(tensor_parallel_size=2, data_parallel_size=16,
                                         pipeline_parallel_size=1, prefill_context_parallel_size=1)
        finegrained = types.SimpleNamespace(oproj_dp_size=2, oproj_communication="all_reduce",
            oproj_native_communication=native,
            lmhead_tensor_parallel_size=2 if native else 16, embedding_tensor_parallel_size=0, mlp_tensor_parallel_size=0)
        config = types.SimpleNamespace(pd_tp_ratio=1, pd_head_ratio=1,
            eplb_config=types.SimpleNamespace(dynamic_eplb=False), finegrained_tp_config=finegrained)
        namespace = {name: None for name in ("_MC2", "_OTP", "_OTP_TOTAL", "_LMTP", "_MLP_TP",
                                             "_EMBED_TP", "_P_TP", "_DYNAMIC_EPLB", "_OPROJ_NATIVE")}
        namespace.update(
            torch=types.SimpleNamespace(arange=torch.arange, device=lambda *args: "npu:0",
                npu=types.SimpleNamespace(current_device=lambda: 0), distributed=types.SimpleNamespace(
                is_initialized=lambda: True, get_world_size=lambda: 32, get_backend=lambda _: "mock")),
            get_world_group=lambda: types.SimpleNamespace(local_rank=0, device_group=None),
            get_ascend_config=lambda: config, init_model_parallel_group=create,
        )
        load_nodes("vllm_ascend/distributed/parallel_state.py",
                   ["model_parallel_initialized", "init_ascend_model_parallel", "destroy_ascend_model_parallel"],
                   namespace)
        def native_init(otp, total, device):
            self.assertIn("lmheadtp", created)
            self.assertIsNot(otp, namespace["_LMTP"])
            self.assertIs(otp, namespace["_OTP"])
            self.assertIs(total, namespace["_OTP_TOTAL"])
            return types.SimpleNamespace(close=lambda: destroyed.append("native"))

        module = types.ModuleType("vllm_ascend.distributed.oproj_communicator")
        module.OProjNativeCommunicator = native_init
        with patch.dict(sys.modules, {module.__name__: module}):
            namespace["init_ascend_model_parallel"](parallel)
        self.assertEqual(created["otp"], [[base, base + 2] for chunk in range(0, 32, 4)
                                          for base in (chunk, chunk + 1)])
        self.assertEqual(created["otp_total"], [list(range(base, base + 4)) for base in range(0, 32, 4)])
        namespace["destroy_ascend_model_parallel"]()
        if native:
            self.assertEqual(destroyed[0], "native")
            self.assertIsNone(namespace["_OPROJ_NATIVE"])
        self.assertEqual(destroyed.count("otp_total"), 1)
        self.assertIsNone(namespace["_OTP_TOTAL"])

    def test_production_group_builder_and_destroy(self):
        self.check_group_builder_and_destroy(False)

    def test_native_otp_never_aliases_same_size_lmhead_group(self):
        self.check_group_builder_and_destroy(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args, remaining = parser.parse_known_args()
    SOURCE_ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
