# SPDX-License-Identifier: Apache-2.0
"""The SDK facade must retain its own native protocol symbol bindings."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/memcache_compat.py"
spec = importlib.util.spec_from_file_location("memcache_compat_under_test", SOURCE)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class TestMemcacheCompat(unittest.TestCase):
    def tearDown(self):
        compat.load_memcache.cache_clear()
        compat._native_handles = ()

    def test_protocol_scope_is_ready_before_facade_import_and_is_cached(self):
        root = Path("/sdk/memcache_hybrid")
        facade = object()
        events = []
        handles = []

        def load(path, mode):
            self.assertEqual(mode, 1 | 8)
            events.append(path)
            handle = object()
            handles.append(handle)
            return handle

        def sdk_import(name):
            if name == "memfabric_hybrid":
                events.append(name)
                return object()
            self.assertEqual(name, "memcache_hybrid")
            self.assertEqual(
                events,
                ["memfabric_hybrid", str(root / "lib/libmf_memcache.so"), str(root / "_pymmc.so")],
            )
            self.assertEqual(compat._native_handles, tuple(handles))
            return facade

        with (
            patch.object(compat, "import_module", side_effect=sdk_import) as imports,
            patch.object(compat.metadata, "distribution", return_value=SimpleNamespace(locate_file=lambda _: root)),
            patch.object(compat.PathFinder, "find_spec", return_value=SimpleNamespace(origin=str(root / "_pymmc.so"))),
            patch.object(compat.ctypes, "CDLL", side_effect=load) as native,
            patch.object(compat, "os", SimpleNamespace(RTLD_NOW=1, RTLD_LOCAL=0, RTLD_DEEPBIND=8)),
        ):
            self.assertIs(compat.load_memcache(), facade)
            self.assertIs(compat.load_memcache(), facade)
            self.assertEqual(native.call_count, 2)
            self.assertEqual(imports.call_count, 2)

    def test_missing_native_extension_fails_without_default_sdk_import(self):
        with (
            patch.object(compat, "import_module") as imports,
            patch.object(
                compat.metadata, "distribution", return_value=SimpleNamespace(locate_file=lambda _: Path("/sdk"))
            ),
            patch.object(compat.PathFinder, "find_spec", return_value=None),
            self.assertRaisesRegex(ImportError, "extension is missing"),
        ):
            compat.load_memcache()
        imports.assert_called_once_with("memfabric_hybrid")


if __name__ == "__main__":
    unittest.main()
