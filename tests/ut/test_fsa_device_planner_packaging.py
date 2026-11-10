# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for rebuilding a package after disabling the FSA planner."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
