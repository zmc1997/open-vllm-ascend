# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a Python package build retains the BMM's HF32 control files."""

import ast
import tempfile
import unittest
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parents[3]


class TestVupBmmPackaging(unittest.TestCase):
    def test_build_py_copies_hf32_control_files(self):
        # Read the actual packaging declaration without executing setup.py's
        # device detection or extension compilation on this CPU-only test.
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        setup = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
        )
        declaration = next(keyword.value for keyword in setup.keywords if keyword.arg == "package_data")
        distribution = Distribution(
            {
                "packages": ["vllm_ascend.ops.triton"],
                "package_dir": {"": str(ROOT)},
                "package_data": ast.literal_eval(declaration),
            }
        )
        distribution.script_name = str(ROOT / "setup.py")
        with tempfile.TemporaryDirectory() as directory:
            command = build_py(distribution)
            command.build_lib = directory
            command.ensure_finalized()
            command.run()
            for filename in ("vup_hf32_control.bc", "vup_hf32_control.cpp"):
                relative = Path("vllm_ascend/ops/triton") / filename
                self.assertEqual((Path(directory) / relative).read_bytes(), (ROOT / relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
