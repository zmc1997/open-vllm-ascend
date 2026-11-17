# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check FSA planner build selection, packaging and lookup without an NPU."""

import ast
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class BuildCommand:
    def run(self):
        pass


def load_build_commands(root, planner_enabled, compile_kernels):
    # Exercise the actual build command hooks without detecting an NPU or
    # starting a native compilation.
    names = {"custom_build_info", "cmake_build_ext", "remove_disabled_fsa_planner"}
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {
        "BUILD_FSA_DEVICE_PLANNER": planner_enabled,
        "FSA_DEVICE_PLANNER_LIBRARY": "libfsa_mapped_recurrent_planner.so",
        "ROOT_DIR": str(root),
        "Path": Path,
        "os": os,
        "build_py": BuildCommand,
        "build_ext": BuildCommand,
        "CMakeExtension": object,
        "envs": SimpleNamespace(COMPILE_CUSTOM_KERNELS=compile_kernels),
        "gen_build_info": Mock(),
    }
    exec(compile(tree, str(ROOT / "setup.py"), "exec"), namespace)
    return namespace


class TestFsaDevicePlannerPackaging(unittest.TestCase):
    def test_default_build_uses_the_normalized_native_target(self):
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        start = next(
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CANN_SOC_VERSION"
        )
        code = compile(ast.Module(body=tree.body[start : start + 3], type_ignores=[]), "setup.py", "exec")
        for soc, native, override, expected in (
            ("ascend910_9391", True, None, True),
            ("ascend910_9362", True, None, True),
            ("910c", True, None, True),
            ("ascend910b1", True, None, False),
            ("ascend950dt_9582", True, None, False),
            ("ascend910_9391", False, None, False),
            ("ascend910_9391", True, False, False),
            ("ascend910_9391", True, True, True),
        ):
            with self.subTest(soc=soc, native=native, override=override):
                env = SimpleNamespace(
                    SOC_VERSION=soc, COMPILE_CUSTOM_KERNELS=native, VLLM_ASCEND_BUILD_FSA_DEVICE_PLANNER=override
                )
                namespace = {"envs": env, "re": re}
                exec(code, namespace)
                self.assertIs(namespace["BUILD_FSA_DEVICE_PLANNER"], expected)

    def test_npu_mode_loads_from_the_package_and_keeps_legacy_override(self):
        spec = importlib.util.spec_from_file_location("planner_envs", ROOT / "vllm_ascend" / "envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        name = "VLLM_ASCEND_FSA_PLANNER_MODE"
        override = "VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY"
        with patch.dict(os.environ, clear=True):
            self.assertEqual(envs.VLLM_ASCEND_FSA_PLANNER_MODE, "cpu")
            self.assertIsNone(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY)
            os.environ[name] = "hccl"
            expected = Path(envs.__file__).resolve().with_name("libfsa_mapped_recurrent_planner.so")
            self.assertEqual(Path(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY), expected)
            os.environ[override] = str(ROOT / "override.so")
            self.assertEqual(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY, str(ROOT / "override.so"))
            os.environ[name] = "cpu"
            self.assertIsNone(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY)
            del os.environ[name]
            self.assertEqual(envs.VLLM_ASCEND_FSA_PLANNER_MODE, "hccl")
            self.assertEqual(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY, str(ROOT / "override.so"))
            os.environ[name] = "invalid"
            with self.assertRaisesRegex(ValueError, name):
                _ = envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY

    def test_notify_mode_uses_the_packaged_library_and_respects_explicit_mode(self):
        spec = importlib.util.spec_from_file_location("planner_envs", ROOT / "vllm_ascend" / "envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        with patch.dict(os.environ, {"VLLM_ASCEND_FSA_PLANNER_MODE": "notify"}, clear=True):
            self.assertTrue(envs.VLLM_ASCEND_FSA_PLAN_IPC)
            self.assertEqual(
                Path(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY),
                ROOT / "vllm_ascend" / "libfsa_mapped_recurrent_planner.so",
            )
            os.environ["VLLM_ASCEND_FSA_PLAN_IPC"] = "1"
            os.environ["VLLM_ASCEND_FSA_PLANNER_MODE"] = "hccl"
            self.assertFalse(envs.VLLM_ASCEND_FSA_PLAN_IPC)
            os.environ["VLLM_ASCEND_FSA_PLANNER_MODE"] = "cpu"
            self.assertFalse(envs.VLLM_ASCEND_FSA_PLAN_IPC)
            self.assertIsNone(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY)
            del os.environ["VLLM_ASCEND_FSA_PLANNER_MODE"]
            self.assertTrue(envs.VLLM_ASCEND_FSA_PLAN_IPC)
            os.environ["VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY"] = str(ROOT / "override.so")
            self.assertEqual(envs.VLLM_ASCEND_FSA_DEVICE_PLANNER_LIBRARY, str(ROOT / "override.so"))

    def test_rebuilding_only_keeps_planner_when_enabled(self):
        for command_name, compile_kernels in (("custom_build_info", False), ("cmake_build_ext", True)):
            for planner_enabled in ((False, True) if compile_kernels else (False,)):
                with self.subTest(command=command_name, planner_enabled=planner_enabled):
                    with tempfile.TemporaryDirectory() as directory:
                        source = Path(directory) / "source"
                        build_lib = Path(directory) / "build"
                        namespace = load_build_commands(source, planner_enabled, compile_kernels)
                        planner_name = namespace["FSA_DEVICE_PLANNER_LIBRARY"]
                        for root in (source, build_lib):
                            package = root / "vllm_ascend"
                            package.mkdir(parents=True)
                            (package / planner_name).write_bytes(b"previous enabled build")
                            (package / "vllm_ascend_C.so").write_bytes(b"other extension")
                        command = namespace[command_name]()
                        command.build_lib = str(build_lib)
                        command.run_command = Mock()
                        command.run()
                        for root in (source, build_lib):
                            package = root / "vllm_ascend"
                            self.assertEqual((package / planner_name).exists(), planner_enabled)
                            self.assertEqual((package / "vllm_ascend_C.so").read_bytes(), b"other extension")


if __name__ == "__main__":
    unittest.main()
