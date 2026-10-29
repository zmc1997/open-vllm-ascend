# SPDX-License-Identifier: Apache-2.0
"""CPU checks for actual per-device resource selection; no NPU imports."""

import argparse
import ast
import configparser
import platform
import sys
import unittest
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/kernel_resources.py"


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("platform-fixture").resolve()
        self.directory = self.root / "data/platform_config"
        self.files = {}
        files = self.files

        class FixtureConfig(configparser.ConfigParser):
            def read(self, path, **kwargs):
                self.read_string(files[Path(path)])
                return [str(path)]

        patched = patch.object(Path, "is_file", lambda path: path in files)
        patched.start()
        self.addCleanup(patched.stop)
        self.properties = MagicMock(
            side_effect=lambda device: SimpleNamespace(
                name=f"chip{device}", cube_core_num=24 if device == 0 else 12, L2_cache_size=201326592
            )
        )
        self.driver = MagicMock(side_effect=lambda device: {"num_vectorcore": 48 if device == 0 else 24})
        namespace = {
            "NamedTuple": NamedTuple,
            "cache": cache,
            "configparser": SimpleNamespace(ConfigParser=FixtureConfig),
            "platform": platform,
            "Path": Path,
            "torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=self.properties)),
            "triton": SimpleNamespace(
                runtime=SimpleNamespace(
                    driver=SimpleNamespace(
                        active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=self.driver))
                    )
                )
            ),
            "envs": SimpleNamespace(ASCEND_HOME_PATH=str(self.root)),
        }
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.get = namespace["get_device_resources"]

    def write_config(self, device, ub, npu_arch=2201):
        self.files[self.directory / f"chip{device}.ini"] = (
            f"[version]\nNpuArch={npu_arch}\n[AICoreSpec]\nub_size={ub}\nl1_size=524288\nl0_a_size=65536\n"
            "l0_b_size=65536\nl0_c_size=131072\n"
        )

    def test_device_capacities_are_read_once_per_device_and_immutable(self):
        self.write_config(0, 196608)
        self.write_config(1, 98304)
        first, second = self.get(0), self.get(1)
        self.assertEqual((first.vector_cores, first.cube_cores, first.ub_bytes), (48, 24, 196608))
        self.assertEqual((second.vector_cores, second.cube_cores, second.ub_bytes), (24, 12, 98304))
        self.assertEqual(
            (first.l1_bytes, first.l0a_bytes, first.l0b_bytes, first.l0c_bytes, first.l2_bytes),
            (524288, 65536, 65536, 131072, 201326592),
        )
        self.assertEqual((first.npu_arch, first.cache_line_bytes), (2201, 512))
        self.assertIs(self.get(0), first)
        self.assertIs(self.get(1), second)
        self.assertEqual(self.properties.call_count, 2)
        self.assertEqual(self.driver.call_count, 2)
        with self.assertRaises(AttributeError):
            first.ub_bytes = 1

    def test_architecture_directory_is_supported_without_capacity_defaults(self):
        self.directory = self.root / (platform.machine() + "-linux") / "data/platform_config"
        self.write_config(0, 123456)
        self.assertEqual(self.get(0).ub_bytes, 123456)
        with self.assertRaisesRegex(RuntimeError, "platform configuration"):
            self.get(1)
        self.write_config(1, 98304)
        path = self.directory / "chip1.ini"
        self.files[path] = self.files[path].replace("ub_size=98304\n", "")
        with self.assertRaises(configparser.NoOptionError):
            self.get(1)

    def test_cache_line_comes_from_cann_architecture_rule(self):
        for architecture, expected in ((2201, 512), (3003, 512), (3113, 512), (2002, 256)):
            self.get.cache_clear()
            self.write_config(0, 196608, architecture)
            self.assertEqual(self.get(0).cache_line_bytes, expected)
        self.get.cache_clear()
        self.files[self.directory / "chip0.ini"] = self.files[self.directory / "chip0.ini"].replace(
            "NpuArch=2002\n", ""
        )
        with self.assertRaises(configparser.NoOptionError):
            self.get(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args, remaining = parser.parse_known_args()
    if args.source is not None:
        SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
