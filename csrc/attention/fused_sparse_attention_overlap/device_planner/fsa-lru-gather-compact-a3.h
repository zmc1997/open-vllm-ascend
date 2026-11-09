#pragma once

#include "kernel_operator.h"

namespace FsaLruCompact {
constexpr uint32_t kWidth = 4096;
constexpr uint32_t kMaskBytes = kWidth / 8;
constexpr uint32_t kGroupWords = 2 * kWidth;

__aicore__ inline void MakeLruIndices(const AscendC::LocalTensor<float>& indices, const AscendC::LocalTensor<int32_t>& scratch)
{
    AscendC::ArithProgression<int32_t>(scratch, 0, 1, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(indices, scratch, AscendC::RoundMode::CAST_ROUND, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
}

// firstTopkByRank is already float32: -1 for evictable, >=0 for hit.
// Set its final lane to >=0 so the reserved LRU slot is copied as the last kept entry.
// Each other lane must represent one valid distinct old LRU slot; the caller resolves
// duplicate token winners separately but marks every matching resident slot as a hit.
// groups[0..4095] holds compact evictable slots; groups[4096..8191] holds compact hits.
__aicore__ inline uint32_t CompactLru(const AscendC::LocalTensor<int32_t>& oldLru,
                                     const AscendC::LocalTensor<float>& firstTopkByRank,
                                     const AscendC::LocalTensor<int32_t>& groups,
                                     const AscendC::LocalTensor<uint8_t>& mask)
{
    const AscendC::GatherMaskParams params{1, 1, 8, 0};
    const auto packedMask = mask.ReinterpretCast<uint32_t>();
    uint64_t evictableCount = 0;
    uint64_t keptCount = 0;
    AscendC::CompareScalar(mask, firstTopkByRank, -1.0f, AscendC::CMPMODE::EQ, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GatherMask(groups, oldLru, packedMask, true, kWidth, params, evictableCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CompareScalar(mask, firstTopkByRank, 0.0f, AscendC::CMPMODE::GE, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GatherMask(groups[kWidth], oldLru, packedMask, true, kWidth, params, keptCount);
    AscendC::PipeBarrier<PIPE_V>();
    return static_cast<uint32_t>(evictableCount);
}

// assignedCount <= evictableCount. Each tensor starts at a 32-byte boundary.
// The output can reuse oldLru after CompactLru; it cannot alias groups, indices or scratch.
// Groups and scratch remain live through the final Gather. The caller owns the V_MTE3 edge.
__aicore__ inline void RotateLru(const AscendC::LocalTensor<int32_t>& out,
                               const AscendC::LocalTensor<int32_t>& groups,
                               const AscendC::LocalTensor<float>& indices,
                               const AscendC::LocalTensor<int32_t>& scratch,
                               const AscendC::LocalTensor<uint8_t>& mask,
                               uint32_t evictableCount, uint32_t assignedCount)
{
    const float assigned = static_cast<float>(static_cast<int32_t>(assignedCount));
    const float evictable = static_cast<float>(static_cast<int32_t>(evictableCount));
    const auto outFloat = out.ReinterpretCast<float>();
    const auto workFloat = scratch.ReinterpretCast<float>();
    constexpr auto selectScalar = AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE;

    // For [0,E-A), [E-A,E), and [E,N), add A, A-E, and N-E respectively.
    AscendC::Duplicate(outFloat, assigned - evictable, kWidth);
    AscendC::CompareScalar(mask, indices, evictable - assigned, AscendC::CMPMODE::GE, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(workFloat, mask, outFloat, assigned, selectScalar, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CompareScalar(mask, indices, evictable, AscendC::CMPMODE::LT, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(outFloat, mask, workFloat, static_cast<float>(static_cast<int32_t>(kWidth)) - evictable, selectScalar, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(workFloat, indices, outFloat, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(out, workFloat, AscendC::RoundMode::CAST_ROUND, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scratch, out, static_cast<int32_t>(sizeof(int32_t)), kWidth);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(out, groups, scratch.ReinterpretCast<uint32_t>(), 0U, kWidth);
    AscendC::PipeBarrier<PIPE_V>();
}
} // namespace FsaLruCompact
