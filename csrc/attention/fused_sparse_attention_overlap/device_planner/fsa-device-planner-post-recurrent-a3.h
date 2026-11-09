#pragma once

#include "fsa-lru-gather-compact-a3.h"
#include "fsa-lru-prefix-a3.h"

namespace FsaLruPost {
using namespace AscendC;
using F32 = LocalTensor<float>;
using I32 = LocalTensor<int32_t>;
using Mask = LocalTensor<uint8_t>;

constexpr uint32_t kWidth = FsaLruCompact::kWidth;
constexpr uint32_t kTopk = 2048;
constexpr uint32_t kMaxToken = 1048576;
constexpr uint32_t kPrefixPad = 8;
constexpr uint32_t kPaddedWidth = kWidth + kPrefixPad;
constexpr uint32_t kMaskWords = FsaLruCompact::kMaskBytes / sizeof(int32_t);
constexpr uint32_t kWorkWords = 3 * kWidth + 2 * kPaddedWidth + 4 * kMaskWords;

struct Result {
    uint32_t assignedCount;
    uint32_t currentLocalSlot;
};

__aicore__ inline void Intersect(const Mask& left, const Mask& right, uint32_t count)
{
    auto bits = left.ReinterpretCast<uint16_t>();
    And(bits, bits, right.ReinterpretCast<uint16_t>(), count / 16);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MakeFlags(const F32& flags, const Mask& mask, const F32& temp, uint32_t count)
{
    Duplicate(temp, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Select(flags, mask, temp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline void GatherAt(const LocalTensor<T>& dst, const LocalTensor<T>& source,
                                const I32& offsets, const F32& index, uint32_t count)
{
    Cast(offsets, index, RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
    Muls(offsets, offsets, static_cast<int32_t>(sizeof(T)), count);
    PipeBarrier<PIPE_V>();
    Gather(dst, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GatherAtInt(const I32& dst, const I32& source,
                                   const I32& offsets, const I32& index, uint32_t count)
{
    Muls(offsets, index, static_cast<int32_t>(sizeof(int32_t)), count);
    PipeBarrier<PIPE_V>();
    Gather(dst, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline uint32_t Compact(const LocalTensor<T>& dst, const LocalTensor<T>& source,
                                   const Mask& mask, uint32_t count)
{
    const GatherMaskParams params{1, 1, 8, 0};
    uint64_t kept = 0;
    GatherMask(dst, source, mask.ReinterpretCast<uint32_t>(), true, count, params, kept);
    PipeBarrier<PIPE_V>();
    return static_cast<uint32_t>(kept);
}

// Publish the next inverse while the old-order evict prefix is still live.
// The clobbered iota UB is not reused until the caller waits MTE3 -> V.
__aicore__ inline void PublishNextInverse(
    const F32& iota, const F32& evictPrefix, const F32& spare, const F32& temp,
    const I32& offsets, const GlobalTensor<int32_t>& oldInverse, const Mask& evictMask, const Mask& auxMask,
    const GlobalTensor<int32_t>& nextInverse, float evictCount, float assigned, bool reset, event_t event)
{
    auto floatOffsets = offsets.ReinterpretCast<float>();
    auto ranks = spare.ReinterpretCast<int32_t>();
    auto physicalRanks = iota.ReinterpretCast<int32_t>();
    // Hit: E + old rank - Eprefix. Evict: Eprefix - 1 - A, rotated by E for
    // the first A evicts. The non-evictable reserved tail stays at rank 4095.
    Sub(spare, iota, evictPrefix, kWidth);
    Adds(temp, evictPrefix, -1.0f - assigned, kWidth);
    CompareScalar(auxMask, evictPrefix, assigned, CMPMODE::LE, kWidth);
    PipeBarrier<PIPE_V>();
    Adds(spare, spare, evictCount, kWidth);
    Adds(floatOffsets, temp, evictCount, kWidth);
    PipeBarrier<PIPE_V>();
    Select(temp, auxMask, floatOffsets, temp, SELMODE::VSEL_TENSOR_TENSOR_MODE, kWidth);
    PipeBarrier<PIPE_V>();
    if (!reset) {
        // The float offset scratch is dead. Stage the old inverse while the
        // final selection and cast compute the new ranks in old LRU order.
        SetFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::V_MTE2>(event);
        DataCopy(offsets, oldInverse, kWidth);
        SetFlag<HardEvent::MTE2_V>(event);
    }
    Select(iota, evictMask, temp, spare, SELMODE::VSEL_TENSOR_TENSOR_MODE, kWidth);
    PipeBarrier<PIPE_V>();
    Cast(ranks, iota, RoundMode::CAST_ROUND, kWidth);
    if (reset) {
        // After reset the old LRU is identity, so no gather is necessary.
        PipeBarrier<PIPE_V>();
        DataCopy(physicalRanks, ranks, kWidth);
    } else {
        WaitFlag<HardEvent::MTE2_V>(event);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), kWidth);
        PipeBarrier<PIPE_V>();
        Gather(physicalRanks, ranks, offsets.ReinterpretCast<uint32_t>(), 0, kWidth);
    }
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopy(nextInverse, physicalRanks, kWidth);
    SetFlag<HardEvent::MTE3_V>(event);
}

// lru and slots are in/out. Inputs are post-reset LRU and physical resident state,
// with lru[4095] == oldInverse[4095] == 4095. The active LRU is a permutation.
// winnerByTopk resolves duplicate resident hits to the highest old LRU rank.
// firstTopkByRank uses -1 for evictable, and lists every hit even for duplicate slots' tokens.
// firstTopkByRank and winnerByTopk are clobbered after their last use. All buffers are
// disjoint and 32-byte aligned. work has kWorkWords words; event is a caller-owned V_S event.
__aicore__ inline Result PlanAfterLookup(const I32& lru, const I32& slots, const I32& topk,
                                         const I32& firstTopkByRank, const I32& winnerByTopk,
                                         const GlobalTensor<int32_t>& oldInverse, const I32& plan, const I32& current,
                                         const I32& work, const GlobalTensor<int32_t>& nextInverse,
                                         uint32_t stable, uint32_t visible, bool reset, event_t event)
{
    const I32 groups = work;
    const F32 temp = work[kWidth].ReinterpretCast<float>();
    const F32 prefixSource = work[2 * kWidth].ReinterpretCast<float>();
    const F32 prefixWork = work[2 * kWidth + kPaddedWidth].ReinterpretCast<float>();
    const F32 a = prefixSource[kPrefixPad];
    const F32 b = prefixWork[kPrefixPad];
    const I32 offsets = work[2 * kWidth + 2 * kPaddedWidth];
    const I32 masks = work[3 * kWidth + 2 * kPaddedWidth];
    const Mask evictMask = masks.ReinterpretCast<uint8_t>();
    const Mask missMask = masks[kMaskWords].ReinterpretCast<uint8_t>();
    const Mask selectMask = masks[2 * kMaskWords].ReinterpretCast<uint8_t>();
    const Mask auxMask = masks[3 * kMaskWords].ReinterpretCast<uint8_t>();
    const F32 iota = firstTopkByRank.ReinterpretCast<float>();
    constexpr auto selectScalar = SELMODE::VSEL_TENSOR_SCALAR_MODE;

    Cast(temp, firstTopkByRank, RoundMode::CAST_ROUND, kWidth);
    PipeBarrier<PIPE_V>();
    CompareScalar(evictMask, temp, -1.0f, CMPMODE::EQ, kWidth);
    FsaLruCompact::MakeLruIndices(iota, offsets);
    CompareScalar(auxMask, iota, static_cast<float>(static_cast<int32_t>(kWidth - 1)), CMPMODE::LT, kWidth);
    PipeBarrier<PIPE_V>();
    Intersect(evictMask, auxMask, kWidth);
    Duplicate(groups, int32_t{0}, 8);  // A defined index zero even when every active slot hits.
    PipeBarrier<PIPE_V>();
    const uint32_t evictCount = Compact(groups, lru, evictMask, kWidth);

    Cast(temp, topk, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    CompareScalar(missMask, temp, 0.0f, CMPMODE::GE, kTopk);
    CompareScalar(auxMask, temp, static_cast<float>(static_cast<int32_t>(visible)), CMPMODE::LT, kTopk);
    PipeBarrier<PIPE_V>();
    Intersect(missMask, auxMask, kTopk);
    Cast(temp, winnerByTopk, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    CompareScalar(auxMask, temp, 0.0f, CMPMODE::LT, kTopk);
    PipeBarrier<PIPE_V>();
    Intersect(missMask, auxMask, kTopk);

    MakeFlags(a, missMask, temp, kTopk);
    const F32 missPrefix = fsa_lru_prefix::InclusiveScan(prefixSource, prefixWork, temp, offsets, kTopk);
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    const uint32_t missCount = static_cast<int32_t>(missPrefix.GetValue(kTopk - 1));
    const uint32_t assigned = missCount < evictCount ? missCount : evictCount;
    const float assignedF = static_cast<float>(static_cast<int32_t>(assigned));
    const float currentToken = static_cast<float>(static_cast<int32_t>(visible)) - 1.0f;
    CompareScalar(selectMask, missPrefix, assignedF, CMPMODE::LE, kTopk);
    PipeBarrier<PIPE_V>();
    Intersect(selectMask, missMask, kTopk);
    Select(temp, selectMask, missPrefix, 1.0f, selectScalar, kTopk);
    PipeBarrier<PIPE_V>();
    Adds(temp, temp, -1.0f, kTopk);
    PipeBarrier<PIPE_V>();
    GatherAt(current, groups, offsets, temp, kTopk);

    Cast(a, current, RoundMode::CAST_ROUND, kTopk);
    Cast(b, winnerByTopk, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    Select(temp, selectMask, a, b, SELMODE::VSEL_TENSOR_TENSOR_MODE, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(current, temp, RoundMode::CAST_ROUND, kTopk);
    Adds(a, temp, 1.0f, kTopk);
    PipeBarrier<PIPE_V>();
    Muls(b, a, -1.0f, kTopk);
    Cast(temp, topk, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    CompareScalar(auxMask, temp, currentToken, CMPMODE::NE, kTopk);
    PipeBarrier<PIPE_V>();
    Intersect(selectMask, auxMask, kTopk);
    Select(temp, selectMask, b, a, SELMODE::VSEL_TENSOR_TENSOR_MODE, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(plan, temp, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();

    // Position is the high part of the score, slot is the low part. The largest
    // eligible score selects the last current-token assignment. All scores are exact f32.
    Cast(a, topk, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    CompareScalar(selectMask, a, currentToken, CMPMODE::EQ, kTopk);
    PipeBarrier<PIPE_V>();
    Cast(a, current, RoundMode::CAST_ROUND, kTopk);
    PipeBarrier<PIPE_V>();
    CompareScalar(auxMask, a, 0.0f, CMPMODE::GE, kTopk);
    Adds(b, iota, 1.0f, kTopk);
    PipeBarrier<PIPE_V>();
    Intersect(selectMask, auxMask, kTopk);
    Muls(b, b, static_cast<float>(static_cast<int32_t>(kWidth)), kTopk);
    PipeBarrier<PIPE_V>();
    Add(temp, b, a, kTopk);
    PipeBarrier<PIPE_V>();
    Select(b, selectMask, temp, 0.0f, selectScalar, kTopk);
    PipeBarrier<PIPE_V>();
    const F32 reduced = offsets.ReinterpretCast<float>();
    ReduceMax(reduced, b, temp, kTopk);
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    const uint32_t score = static_cast<int32_t>(reduced.GetValue(0));
    const uint32_t currentLocalSlot = score == 0 ? kWidth - 1 : score % kWidth;

    const F32 compactMissPos = winnerByTopk.ReinterpretCast<float>();
    Duplicate(compactMissPos, 0.0f, kTopk);  // Index zero is defined even if there are no misses.
    PipeBarrier<PIPE_V>();
    Compact(compactMissPos, iota, missMask, kTopk);
    MakeFlags(a, evictMask, temp, kWidth);
    const F32 evictPrefix = fsa_lru_prefix::InclusiveScan(prefixSource, prefixWork, temp, offsets, kWidth);
    // Width 4096 has 12 scan passes, leaving prefixSource live and b free.
    static_assert(kWidth == 4096);
    PublishNextInverse(iota, evictPrefix, b, temp, offsets, oldInverse, evictMask, auxMask,
                       nextInverse, static_cast<float>(static_cast<int32_t>(evictCount)), assignedF, reset, event);
    CompareScalar(selectMask, evictPrefix, assignedF, CMPMODE::LE, kWidth);
    PipeBarrier<PIPE_V>();
    Intersect(selectMask, evictMask, kWidth);
    Select(temp, selectMask, evictPrefix, 1.0f, selectScalar, kWidth);
    PipeBarrier<PIPE_V>();
    Adds(temp, temp, -1.0f, kWidth);
    PipeBarrier<PIPE_V>();

    GatherAt(a, compactMissPos, offsets, temp, kWidth);
    GatherAt(b.ReinterpretCast<int32_t>(), topk, offsets, a, kWidth);
    GatherAtInt(a.ReinterpretCast<int32_t>(), slots, offsets, lru, kWidth);
    WaitFlag<HardEvent::MTE3_V>(event);
    FsaLruCompact::MakeLruIndices(iota, offsets);

    // The kept partition is formed last. It aliases the temporary vector.
    Not(auxMask.ReinterpretCast<uint16_t>(), evictMask.ReinterpretCast<uint16_t>(), kWidth / 16);
    PipeBarrier<PIPE_V>();
    Compact(groups[kWidth], lru, auxMask, kWidth);
    FsaLruCompact::RotateLru(lru, groups, iota, offsets, auxMask, evictCount, assigned);

    Cast(temp, a.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, kWidth);
    PipeBarrier<PIPE_V>();
    CompareScalar(missMask, temp, static_cast<float>(static_cast<int32_t>(stable)), CMPMODE::GE, kWidth);
    CompareScalar(auxMask, temp, static_cast<float>(static_cast<int32_t>(kMaxToken)), CMPMODE::LT, kWidth);
    PipeBarrier<PIPE_V>();
    Intersect(missMask, auxMask, kWidth);
    CompareScalar(auxMask, iota, static_cast<float>(static_cast<int32_t>(kWidth - 1)), CMPMODE::LT, kWidth);
    PipeBarrier<PIPE_V>();
    Intersect(missMask, auxMask, kWidth);
    const auto suffixBits = missMask.ReinterpretCast<uint16_t>();
    Not(suffixBits, suffixBits, kWidth / 16);
    PipeBarrier<PIPE_V>();
    // Select addresses, not token values. The final int32 Gather preserves arbitrary
    // old invalid/reserved bits that would be rounded by a float round-trip.
    const I32 rankSources = prefixSource.ReinterpretCast<int32_t>();
    const F32 choice = groups.ReinterpretCast<float>();
    const F32 selected = offsets.ReinterpretCast<float>();
    Duplicate(rankSources, int32_t{-1}, kPrefixPad);
    Adds(temp, iota, static_cast<float>(static_cast<int32_t>(kPrefixPad)), kWidth);
    PipeBarrier<PIPE_V>();
    Select(choice, missMask, temp, 0.0f, selectScalar, kWidth);
    PipeBarrier<PIPE_V>();
    Adds(temp, iota, static_cast<float>(static_cast<int32_t>(kPaddedWidth + kPrefixPad)), kWidth);
    PipeBarrier<PIPE_V>();
    Select(selected, selectMask, temp, choice, SELMODE::VSEL_TENSOR_TENSOR_MODE, kWidth);
    PipeBarrier<PIPE_V>();
    GatherAt(temp.ReinterpretCast<int32_t>(), rankSources, choice.ReinterpretCast<int32_t>(), selected, kWidth);
    if (reset) {
        DataCopy(slots, temp.ReinterpretCast<int32_t>(), kWidth);
        PipeBarrier<PIPE_V>();
    } else {
        // As in the validated low-UB gate, load the inverse after the final
        // ranked-token gather frees its offset scratch.
        const I32 inverse = choice.ReinterpretCast<int32_t>();
        SetFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::V_MTE2>(event);
        DataCopy(inverse, oldInverse, kWidth);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        GatherAtInt(slots, temp.ReinterpretCast<int32_t>(), offsets, inverse, kWidth);
    }
    return {assigned, currentLocalSlot};
}
} // namespace FsaLruPost
