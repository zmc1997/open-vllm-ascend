# SPDX-License-Identifier: Apache-2.0
"""Keep Memcache's native protocol bindings separate from MemFabric's."""

import ctypes
import os
from functools import cache
from importlib import import_module, metadata
from importlib.machinery import PathFinder

_native_handles = ()


@cache
def load_memcache():
    # Memcache's facade also imports MemFabric. Load it first, then bind both
    # Memcache binaries privately before the facade imports their handlers.
    import_module("memfabric_hybrid")
    root = metadata.distribution("memcache_hybrid").locate_file("memcache_hybrid")
    extension = PathFinder.find_spec("_pymmc", [str(root)])
    if extension is None:
        raise ImportError("The memcache_hybrid _pymmc extension is missing")
    mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
    global _native_handles
    _native_handles = tuple(
        ctypes.CDLL(str(path), mode=mode) for path in (root / "lib/libmf_memcache.so", extension.origin)
    )
    return import_module("memcache_hybrid")
