#ifndef FSA_LRU_RANKED_A3_H
#define FSA_LRU_RANKED_A3_H

#include "kernel_operator.h"

namespace FsaLruRanked {
using namespace AscendC;

constexpr int kTopk = 2048;
constexpr int kCapacity = 4096;

// Repair arbitrary hardware tie order. The lookup returns the first rank of the
// equal-key run, so each rank needs the minimum original position to its right
// within that run. Pointer doubling handles duplicate runs without scalar walks.
__aicore__ inline void ResolveTopkTies(const LocalTensor<float>& work, int visible)
{
    LocalTensor<float> keys = work[2 * kTopk];
    auto sortedPositions = keys[kTopk].ReinterpretCast<int32_t>();
    LocalTensor<float> minimum = work[4 * kTopk];
    LocalTensor<float> neighborPositions = minimum[kTopk];
    auto byteOffsets = work[4 * kTopk + kCapacity].ReinterpretCast<int32_t>();
    LocalTensor<float> neighborKeys = work[4 * kTopk + kCapacity + kTopk];
    LocalTensor<float> temporary = work[4 * kTopk + 2 * kCapacity];
    auto mask = work[4 * kTopk + 4 * kCapacity].ReinterpretCast<uint8_t>();
    auto validMask = mask[kTopk / 8];

    Cast(minimum, sortedPositions, RoundMode::CAST_ROUND, kTopk);
    ArithProgression<int32_t>(byteOffsets, 4, 4, kTopk);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    byteOffsets.SetValue(kTopk - 1, 0);
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
    Gather(neighborKeys, keys, byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
    CompareScalar(validMask, keys, 0.0f, CMPMODE::GE, kTopk);
    PipeBarrier<PIPE_V>();
    Compare(mask, neighborKeys, keys, CMPMODE::EQ, kTopk);
    PipeBarrier<PIPE_V>();
    auto mask16 = mask.ReinterpretCast<uint16_t>();
    And(mask16, mask16, validMask.ReinterpretCast<uint16_t>(), kTopk / 16);
    CompareScalar(validMask, keys, static_cast<float>(visible), CMPMODE::LT, kTopk);
    PipeBarrier<PIPE_V>();
    And(mask16, mask16, validMask.ReinterpretCast<uint16_t>(), kTopk / 16);
    PipeBarrier<PIPE_V>();
    GatherMaskParams params{1, 1, 8, 0};
    uint64_t duplicates = 0;
    GatherMask(temporary, keys, mask.ReinterpretCast<uint32_t>(), true, kTopk, params, duplicates);
    PipeBarrier<PIPE_V>();
    if (duplicates == 0) {
        return;
    }

    for (int step = 1; step < kTopk; step *= 2) {
        ArithProgression<int32_t>(byteOffsets, 4 * step, 4, kTopk);
        PipeBarrier<PIPE_V>();
        Cast(temporary, byteOffsets, RoundMode::CAST_ROUND, kTopk);
        PipeBarrier<PIPE_V>();
        Mins(temporary, temporary, static_cast<float>(4 * (kTopk - 1)), kTopk);
        PipeBarrier<PIPE_V>();
        Cast(byteOffsets, temporary, RoundMode::CAST_ROUND, kTopk);
        PipeBarrier<PIPE_V>();
        Gather(neighborKeys, keys, byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
        Gather(neighborPositions, minimum, byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
        PipeBarrier<PIPE_V>();
        Compare(mask, neighborKeys, keys, CMPMODE::EQ, kTopk);
        Min(neighborPositions, neighborPositions, minimum, kTopk);
        PipeBarrier<PIPE_V>();
        Select(minimum, mask, neighborPositions, minimum, SELMODE::VSEL_TENSOR_TENSOR_MODE, kTopk);
        PipeBarrier<PIPE_V>();
    }
}

// Convert the existing physical slot lookup to first original TopK positions in
// old LRU order. The caller supplies a valid LRU permutation, with slot 4095 at
// the reserved tail. The only externally visible output is hitPositionByRank.
__aicore__ inline void RankSlotHits(const LocalTensor<int32_t>& lru,
                                    const LocalTensor<int32_t>& hitPositionByRank,
                                    const LocalTensor<float>& work, int stable, int visible)
{
    ResolveTopkTies(work, visible);
    LocalTensor<float> slotKeys = work;
    LocalTensor<float> minimum = work[4 * kTopk];
    LocalTensor<float> floatOffsets = work[4 * kTopk + kCapacity];
    LocalTensor<float> positions = work[4 * kTopk + 2 * kCapacity];
    auto offsets = work[4 * kTopk + 3 * kCapacity].ReinterpretCast<int32_t>();
    auto mask = work[4 * kTopk + 4 * kCapacity].ReinterpretCast<uint8_t>();
    Cast(floatOffsets, offsets, RoundMode::CAST_ROUND, kCapacity);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, floatOffsets, 0.0f, CMPMODE::GE, kCapacity);
    Maxs(floatOffsets, floatOffsets, 0.0f, kCapacity);
    PipeBarrier<PIPE_V>();
    Cast(offsets, floatOffsets, RoundMode::CAST_ROUND, kCapacity);
    PipeBarrier<PIPE_V>();
    Gather(positions, minimum, offsets.ReinterpretCast<uint32_t>(), 0, kCapacity);
    PipeBarrier<PIPE_V>();
    Select(positions, mask, positions, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kCapacity);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, slotKeys, 0.0f, CMPMODE::GE, kCapacity);
    PipeBarrier<PIPE_V>();
    Select(positions, mask, positions, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kCapacity);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, slotKeys, static_cast<float>(stable < visible ? stable : visible), CMPMODE::LT, kCapacity);
    PipeBarrier<PIPE_V>();
    Select(positions, mask, positions, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kCapacity);
    Muls(offsets, lru, 4, kCapacity);
    PipeBarrier<PIPE_V>();
    Gather(floatOffsets, positions, offsets.ReinterpretCast<uint32_t>(), 0, kCapacity);
    PipeBarrier<PIPE_V>();
    Cast(hitPositionByRank, floatOffsets, RoundMode::CAST_ROUND, kCapacity);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    hitPositionByRank.SetValue(kCapacity - 1, -1);
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
}
} // namespace FsaLruRanked

#endif
