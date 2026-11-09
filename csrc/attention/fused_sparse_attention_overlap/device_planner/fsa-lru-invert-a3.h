#ifndef FSA_LRU_INVERT_A3_H
#define FSA_LRU_INVERT_A3_H

#include "kernel_operator.h"

namespace FsaLruInvert {
using namespace AscendC;

constexpr int kTopk = 2048;
constexpr int kCapacity = 4096;
constexpr int kWorkWords = 4 * kCapacity + 4 * kTopk + kTopk / 16;

// Input positions are in old LRU order, with -1 for nonhits (including the reserved
// tail). Output is the winning physical slot at every TopK position, or -1.
// A composite key is exactly representable as FP32 and makes the latest old LRU
// rank win even if every resident slot has the same token.
__aicore__ inline void InvertLruHits(const LocalTensor<int32_t>& hitPositionByRank,
                                     const LocalTensor<int32_t>& lru,
                                     const LocalTensor<int32_t>& winnerSlotByTopk,
                                     const LocalTensor<int32_t>& workWords)
{
    LocalTensor<float> work = workWords.ReinterpretCast<float>();
    LocalTensor<float> sortA = work;
    LocalTensor<float> sortB = work[2 * kCapacity];
    auto keyWords = sortA.ReinterpretCast<int32_t>();
    auto ranks = sortB.ReinterpretCast<int32_t>();
    ArithProgression<int32_t>(ranks, 0, 1, kCapacity);
    Muls(keyWords, hitPositionByRank, kCapacity, kCapacity);
    PipeBarrier<PIPE_V>();
    Add(keyWords, keyWords, ranks, kCapacity);
    PipeBarrier<PIPE_V>();
    Cast(sortA, keyWords, RoundMode::CAST_ROUND, kCapacity);
    DataCopy(sortA[kCapacity].ReinterpretCast<int32_t>(), lru, kCapacity);
    PipeBarrier<PIPE_V>();
    Sort32(sortB, sortA, sortA[kCapacity].ReinterpretCast<uint32_t>(), kCapacity / 32);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> from = sortB, to = sortA;
    for (int tile = 32; tile < kCapacity; tile *= 4) {
        const bool lastTwo = tile * 2 == kCapacity;
        MrgSort4Info params;
        for (int i = 0; i < 4; ++i) {
            params.elementLengths[i] = tile;
        }
        params.ifExhaustedSuspension = false;
        params.validBit = lastTwo ? 0b0011 : 0b1111;
        params.repeatTimes = lastTwo ? 1 : kCapacity / (4 * tile);
        MrgSortSrcList<float> inputs;
        inputs.src1 = from;
        inputs.src2 = from[2 * tile];
        inputs.src3 = lastTwo ? from : from[4 * tile];
        inputs.src4 = lastTwo ? from : from[6 * tile];
        MrgSort<float>(to, inputs, params);
        PipeBarrier<PIPE_V>();
        auto old = from;
        from = to;
        to = old;
    }
    // Four merge passes leave sortB as source and sortA available for extraction.
    auto sortedSlots = sortA[kCapacity].ReinterpretCast<int32_t>();
    Extract(sortA, sortedSlots.ReinterpretCast<uint32_t>(), from, kCapacity / 32);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> query = sortB;
    LocalTensor<float> offsets = work[4 * kCapacity];
    LocalTensor<float> candidate = work[4 * kCapacity + kTopk];
    LocalTensor<float> gathered = work[4 * kCapacity + 2 * kTopk];
    auto byteOffsets = work[4 * kCapacity + 3 * kTopk].ReinterpretCast<int32_t>();
    auto upperMask = work[4 * kCapacity + 4 * kTopk].ReinterpretCast<uint8_t>();
    auto lowerMask = upperMask[kTopk / 8];
    ArithProgression<int32_t>(byteOffsets, kCapacity - 1, kCapacity, kTopk);
    Duplicate(offsets, -4.0f, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(query, byteOffsets, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    for (int step = kCapacity / 2; step > 0; step >>= 1) {
        Adds(candidate, offsets, static_cast<float>(4 * step), kTopk);
        PipeBarrier<PIPE_V>();
        Cast(byteOffsets, candidate, RoundMode::CAST_ROUND, kTopk);
        PipeBarrier<PIPE_V>();
        Gather(gathered, sortA, byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
        PipeBarrier<PIPE_V>();
        Compare(upperMask, gathered, query, CMPMODE::GT, kTopk);
        PipeBarrier<PIPE_V>();
        Select(gathered, upperMask, candidate, offsets, SELMODE::VSEL_TENSOR_TENSOR_MODE, kTopk);
        PipeBarrier<PIPE_V>();
        auto old = offsets;
        offsets = gathered;
        gathered = old;
    }
    Adds(candidate, offsets, 4.0f, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(byteOffsets, candidate, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    Gather(gathered, sortA, byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
    PipeBarrier<PIPE_V>();
    Compare(upperMask, gathered, query, CMPMODE::LE, kTopk);
    Adds(query, query, static_cast<float>(1 - kCapacity), kTopk);
    PipeBarrier<PIPE_V>();
    Compare(lowerMask, gathered, query, CMPMODE::GE, kTopk);
    Gather(gathered.ReinterpretCast<int32_t>(), sortedSlots,
           byteOffsets.ReinterpretCast<uint32_t>(), 0, kTopk);
    PipeBarrier<PIPE_V>();
    auto mask = upperMask.ReinterpretCast<uint16_t>();
    And(mask, mask, lowerMask.ReinterpretCast<uint16_t>(), kTopk / 16);
    Cast(candidate, gathered.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    Select(gathered, upperMask, candidate, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(winnerSlotByTopk, gathered, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
}
} // namespace FsaLruInvert

#endif
