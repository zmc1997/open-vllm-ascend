# SPDX-License-Identifier: Apache-2.0
"""CPU regression for GLM-5.2's gate -> weight loading -> internal router contract.

Run directly with Python; ASCEND_SOURCE_ROOT can point to a checkout or site-packages.
Only the real methods under test are loaded, avoiding unavailable NPU imports.
"""

import __future__

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get("ASCEND_SOURCE_ROOT", Path(__file__).resolve().parents[3]))


def load_class(relative, name, methods, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    tree.body = [cls]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class Weight:
    def __init__(self, dtype="float32"):
        self.dtype = dtype
        self.data = self

    def to(self, dtype):
        return Weight(dtype)


class GateBase:
    def __init__(self, **kwargs):
        self.prefix = kwargs["prefix"]
        self.weight = Weight(kwargs["params_dtype"])
        self.quant_method = LinearBase()


class LinearBase:
    def process_weights_after_loading(self, layer):
        pass


class TestGlmGateSchedule(unittest.TestCase):
    def make_gate(self, enabled, bias=False):
        namespace = {
            "torch": SimpleNamespace(float32="float32"),
            "UnquantizedLinearMethod": LinearBase,
            "_should_keep_nd_for_310p_weight": lambda weight: False,
            "maybe_trans_nz": lambda weight: weight,
        }
        method = load_class(
            "vllm_ascend/ops/linear.py",
            "AscendUnquantizedLinearMethod",
            {"process_weights_after_loading"},
            namespace,
        )
        namespace.update(
            GateLinear=GateBase,
            AscendUnquantizedLinearMethod=method,
            get_ascend_config=lambda: SimpleNamespace(fused_mc2_shared_schedule=enabled),
        )
        cls = load_class("vllm_ascend/ops/fused_moe/gate_linear.py", "AscendGateLinear", {"__init__"}, namespace)
        return cls(6144, 256, bias=bias, prefix="model.layers.0.mlp.gate")

    def load_weights(self, gate):
        # Match the model loader: invoke the method selected by the constructor.
        gate.quant_method.process_weights_after_loading(gate)

    def internal_router(self, gate, enabled=True, shared=True):
        cls = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            {"is_internal_router"},
            {"MoERunner": object},
        )
        runner = cls()
        runner.gate = gate
        runner.fused_mc2_shared_schedule = enabled
        runner._shared_experts = object() if shared else None
        return runner.is_internal_router

    def test_opt_in_enables_internal_router_after_loading(self):
        gate = self.make_gate(True)
        self.assertFalse(self.internal_router(gate))
        self.load_weights(gate)
        self.assertTrue(self.internal_router(gate))
        self.assertEqual(gate.weight_fp32.dtype, "float32")

    def test_disabled_retains_external_router(self):
        gate = self.make_gate(False)
        self.assertIs(type(gate.quant_method), LinearBase)
        self.load_weights(gate)
        self.assertFalse(self.internal_router(gate))

    def test_bias_gate_is_not_moved_into_biasless_internal_router(self):
        gate = self.make_gate(True, bias=True)
        self.assertIs(type(gate.quant_method), LinearBase)
        self.load_weights(gate)
        self.assertFalse(self.internal_router(gate))

    def test_no_shared_experts_keeps_gate_in_model_forward(self):
        gate = self.make_gate(True)
        self.load_weights(gate)
        self.assertFalse(self.internal_router(gate, shared=False))


if __name__ == "__main__":
    unittest.main()
