// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#define ASCENDC_CUBE_ONLY
#include "kernel_operator_mm_intf.h"

// Match TransposeBatchMatMul's SetHF32(true, 1) / SetHF32(false, 0).
// These side effects run on the current Cube core inside the Triton kernel.
extern "C" {
__aicore__ __attribute__((always_inline)) void _mlir_ciface_vup_hf32_enable() {
  AscendC::SetHF32Mode(true);
  AscendC::SetHF32TransMode(true);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_vup_hf32_disable() {
  AscendC::SetHF32Mode(false);
  AscendC::SetHF32TransMode(false);
}
}
