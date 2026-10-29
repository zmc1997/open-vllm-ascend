# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-device resources used by the independent Triton operator launchers."""

import configparser
import platform
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch
from vllm.triton_utils import triton

from vllm_ascend import envs

# Ascend ND DMA blocks are 32 bytes; this is an instruction alignment, not a tile.
DMA_ALIGNMENT_BYTES = 32


class DeviceResources(NamedTuple):
    vector_cores: int
    cube_cores: int
    ub_bytes: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l2_bytes: int
    npu_arch: int
    cache_line_bytes: int


@cache
def get_device_resources(device_index: int) -> DeviceResources:
    """Read the target device and CANN platform once, without tensor transfers."""
    properties = torch.npu.get_device_properties(device_index)
    driver_properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    root = Path(envs.ASCEND_HOME_PATH)
    relative = Path("data/platform_config") / (properties.name + ".ini")
    paths = (root / relative, root / (platform.machine() + "-linux") / relative)
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise RuntimeError(f"CANN platform configuration is required for Triton tiling: {properties.name}")
    config = configparser.ConfigParser()
    config.read(path)
    npu_arch = config.getint("version", "NpuArch")
    return DeviceResources(
        vector_cores=driver_properties["num_vectorcore"],
        cube_cores=properties.cube_core_num,
        ub_bytes=config.getint("AICoreSpec", "ub_size"),
        l1_bytes=config.getint("AICoreSpec", "l1_size"),
        l0a_bytes=config.getint("AICoreSpec", "l0_a_size"),
        l0b_bytes=config.getint("AICoreSpec", "l0_b_size"),
        l0c_bytes=config.getint("AICoreSpec", "l0_c_size"),
        l2_bytes=properties.L2_cache_size,
        npu_arch=npu_arch,
        # CANN v9.1 AscendQuantV2: DAV_2201/3003/3113 use the V220
        # 512-byte cache-line partition; the other architectures use 256.
        # These are platform rules, not measured tile-size preferences.
        cache_line_bytes=512 if npu_arch in (2201, 3003, 3113) else 256,
    )
