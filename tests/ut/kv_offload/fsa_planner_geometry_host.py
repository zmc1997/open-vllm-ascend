# SPDX-License-Identifier: Apache-2.0
"""CPU-only production NPU planner semantics against the original CPU planner.

This compiles the production Ascend C source with checked host vector/DMA shims.
The oracle is the unchanged CPU planner source, not a copied planning algorithm.
This does not establish NPU instruction, event, graph, or performance correctness.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

WRAPPER = r"""
extern "C" HOST_EXPORT void plan_batch(void* req,void* topk,void* stable,void* visible,
    void* last,void* banks,void* state,void* map,void* payload,uint32_t rows,
    uint32_t physical,uint32_t page,uint32_t k,uint32_t c,uint32_t ub,int32_t maximum) {
    MapStableRows((GM_ADDR)req,(GM_ADDR)last,(GM_ADDR)banks,(GM_ADDR)map,rows,physical,page);
    const FsaLruGeometry geometry(k,c,ub);
    for(uint32_t row=0;row<rows;++row)
        PlanRow((GM_ADDR)topk,(GM_ADDR)stable,(GM_ADDR)visible,(GM_ADDR)state,
                (GM_ADDR)map,(GM_ADDR)payload,row,rows,physical,k,c,
                geometry.topkWidth,geometry.tile,geometry.workWords,geometry.lookupMaskWords,maximum,row+1<rows);
}
extern "C" HOST_EXPORT void token_keys(int32_t* input,float* output,float* scalar,uint32_t n) {
    std::vector<int32_t> a(n),b(n);
    AscendC::LocalTensor<int32_t> in{reinterpret_cast<uint8_t*>(input),n*4U};
    AscendC::LocalTensor<float> out{reinterpret_cast<uint8_t*>(output),n*4U};
    AscendC::LocalTensor<int32_t> ta{reinterpret_cast<uint8_t*>(a.data()),n*4U};
    AscendC::LocalTensor<int32_t> tb{reinterpret_cast<uint8_t*>(b.data()),n*4U};
    FsaLruVector::TokenKeys(out,in,ta,tb,n);
    for(uint32_t i=0;i<n;++i)scalar[i]=FsaLruVector::TokenKey(input[i]);
}
extern "C" HOST_EXPORT void sort_topk(int32_t* input,float* keys,float* first,uint32_t n) {
    std::vector<float> scratch(4*n+32);
    std::vector<uint8_t> mask((n+7)/8+32);
    AscendC::LocalTensor<int32_t> in{reinterpret_cast<uint8_t*>(input),n*4U};
    AscendC::LocalTensor<float> k{reinterpret_cast<uint8_t*>(keys),n*4U};
    AscendC::LocalTensor<float> f{reinterpret_cast<uint8_t*>(first),n*4U};
    AscendC::LocalTensor<float> w{reinterpret_cast<uint8_t*>(scratch.data()),scratch.size()*4};
    AscendC::LocalTensor<uint8_t> m{mask.data(),mask.size()};
    FsaLruVector::SortTopk(in,k,f,w,m,n);
}
extern "C" HOST_EXPORT void publish(void* payload,void* membership,void* metadata,
    uint32_t rows,uint32_t k) {
    for(uint32_t row=0;row<rows;++row)
        PublishRows((GM_ADDR)payload,(GM_ADDR)membership,(GM_ADDR)metadata,row,rows,k,row+1<rows);
}
"""

CPU_WRAPPER = r"""
extern "C" HOST_EXPORT void cpu_batch(void* req,void* last,void* topk,void* stable,void* visible,
    void* state,void* mapping,void* plan,uint32_t rows,uint32_t physical,uint32_t k,uint32_t c,
    int32_t maximum) {
    // Scratch belongs only to the independently compiled CPU implementation.
    static std::vector<int32_t> marks,positions;
    marks.assign(maximum,0);positions.assign(maximum,-1);
    std::vector<int32_t> current(rows*k,-1),misses(rows),tokens(rows*k),slots(rows*k);
    std::vector<int32_t> workspace(3*c),missPositions(k);
    int32_t epoch=0;
    auto addr=[](void* p){return reinterpret_cast<uintptr_t>(p);};
    lru_resident_compact_impl(addr(req),addr(last),addr(topk),addr(stable),addr(state),
        addr(static_cast<int32_t*>(state)+physical*c),addr(current.data()),addr(misses.data()),
        addr(tokens.data()),addr(slots.data()),addr(marks.data()),addr(positions.data()),
        addr(workspace.data()),addr(missPositions.data()),addr(&epoch),rows,k,c,maximum,1,1,
        addr(plan),k+16,addr(mapping),physical,addr(visible));
}
"""


class Guarded:
    """Canaries cover every shared buffer, including unpublished graph lanes."""

    def __init__(self, shape, dtype=np.int32, fill=0):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.canary = 12345
        self.storage = np.full(int(np.prod(self.shape)) + 32, self.canary, dtype=dtype)
        self.data = self.storage[16:-16].reshape(self.shape)
        self.data.fill(fill)

    @property
    def ptr(self):
        return ctypes.c_void_p(self.data.ctypes.data)

    def check(self):
        assert np.all(self.storage[:16] == self.canary)
        assert np.all(self.storage[-16:] == self.canary)


def build(source, cpu_source, work, compiler):
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("fixtures") / "fsa_planner_host_shim.h", work / "kernel_operator.h")
    (work / "acl").mkdir(exist_ok=True)
    (work / "acl" / "acl.h").write_text("#pragma once\n", encoding="utf-8")
    device = source.read_text(encoding="utf-8").split('extern "C" __global__')[0]
    cpu = cpu_source.read_text(encoding="utf-8")
    begin = cpu.index("FORCE_INLINE bool is_valid_lru_resident_token")
    end = cpu.index("HOT_FUNCTION void lru_resident_compact(", begin)
    preamble = """
#define FORCE_INLINE inline
#define HOT_FUNCTION
#define RESTRICT
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
constexpr int32_t EPOCH_RESET_THRESHOLD=1<<30;
constexpr int16_t DIRECT_SELECTION_LAYOUT_MARKER=0x5A44;
inline int choose_lru_resident_threads(int,int,int){return 1;}
inline int omp_get_thread_num(){return 0;}
"""
    cpp = work / "geometry.cpp"
    cpp.write_text(device + WRAPPER + preamble + cpu[begin:end] + CPU_WRAPPER, encoding="utf-8")
    library = work / ("geometry.dll" if os.name == "nt" else "geometry.so")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-g",
            "-shared",
            "-fPIC",
            "-I",
            str(work),
            "-I",
            str(source.parent),
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library.resolve()))
    lib.plan_batch.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 6 + [ctypes.c_int32]
    lib.cpu_batch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 4 + [ctypes.c_int32]
    lib.token_keys.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32]
    lib.sort_topk.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32]
    lib.publish.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 2
    return lib


def check_sequence(lib, k, c, physical, logical_counts, maximum, ub, page, seed, seeded_state=False):
    rng = np.random.default_rng(seed)
    state = Guarded((physical, 2, 2, c))
    cpu_state = Guarded((2, physical, c))
    last = Guarded(physical, np.int64, -1)
    cpu_last = Guarded(physical, np.int64, -1)
    banks = Guarded(physical)
    if seeded_state:
        last.data[:] = np.arange(physical, dtype=np.int64) + 100
        cpu_last.data[:] = last.data
    for row in range(physical):
        slots = rng.choice(np.array([-1, 3, 7, 19, maximum - 4, maximum + 3], np.int32), c)
        lru = np.r_[rng.permutation(c - 1), c - 1].astype(np.int32)
        if seeded_state:
            slots[:5] = [3, 3, 19, -927, maximum + 3]
            lru[:-1] = np.arange(c - 2, -1, -1)
        slots[-1] = -1234567
        cpu_state.data[0, row] = slots
        cpu_state.data[1, row] = lru
        for bank in range(2):
            state.data[row, bank, 0] = slots[lru]
            state.data[row, bank, 1] = lru
    wire = ((k + 17) // 2 + 7) // 8 * 8
    words = wire + 8
    # Reuse graph-specific buffers while planner state stays live across graph sizes.
    graph_buffers = {}
    checks = 0
    for step, rows in enumerate(logical_counts):
        if rows not in graph_buffers:
            graph_buffers[rows] = (
                Guarded((rows, words), fill=-987),
                Guarded(4 * physical, fill=-777),
                Guarded((rows, 16400), np.int16, 12345),
                Guarded(rows + 10, fill=12345),
            )
        payload, mapping, membership, metadata = graph_buffers[rows]
        req = Guarded(rows, np.int64)
        current_ids = np.arange(rows, dtype=np.int64) + 100
        if step % 3 == 1:
            current_ids = current_ids[::-1].copy()
        if step % 4 == 2:
            current_ids[::3] += 1000  # New requests force reset/replacement.
        if rows > 4 and step % 5 == 3:
            current_ids[2:4] = current_ids[0]  # Duplicate IDs use distinct physical rows.
        req.data[:] = current_ids
        visible_value = maximum - 2 if step % 3 else min(21, maximum)
        stable_value = visible_value - (0 if step % 2 else 3)
        if step == 6:
            stable_value, visible_value = -5, maximum + 5
        if step == 7:
            stable_value, visible_value = maximum + 5, -1
        stable = Guarded(rows, fill=stable_value)
        visible = Guarded(rows, fill=visible_value)
        topk = Guarded((rows, k))
        tokens = np.array(
            [-7, -1, 3, 7, 19, visible_value - 1, visible_value, maximum - 4, maximum - 3, maximum + 1], dtype=np.int32
        )
        topk.data[:] = rng.choice(tokens, (rows, k))
        if seeded_state and step == 0:
            # No misses overwrite the seeded slots: lookup and invalidation
            # must consume the pre-existing state without resetting the row.
            topk.data.fill(-1)
            topk.data[:, 0] = 3
        if step == 1:
            topk.data.fill(visible_value - 1)
        if step == 4:
            topk.data[:] = rng.choice(np.array([3, 7, 19], np.int32), (rows, k))
        cpu_map = Guarded(3 * physical, fill=-777)
        cpu_plan = Guarded((rows, k + 16), np.int16, fill=-999)
        lib.cpu_batch(
            req.ptr,
            cpu_last.ptr,
            topk.ptr,
            stable.ptr,
            visible.ptr,
            cpu_state.ptr,
            cpu_map.ptr,
            cpu_plan.ptr,
            rows,
            physical,
            k,
            c,
            maximum,
        )
        lib.plan_batch(
            req.ptr,
            topk.ptr,
            stable.ptr,
            visible.ptr,
            last.ptr,
            banks.ptr,
            state.ptr,
            mapping.ptr,
            payload.ptr,
            rows,
            physical,
            page,
            k,
            c,
            ub,
            maximum,
        )
        context = (k, c, physical, rows, maximum, ub, page, step)
        np.testing.assert_array_equal(last.data, cpu_last.data, err_msg=str((context, "requests")))
        np.testing.assert_array_equal(mapping.data[:rows], cpu_map.data[:rows], err_msg=str((context, "mapping")))
        actual_plan = payload.data[:, :wire].copy().view(np.int16).reshape(rows, 2 * wire)[:, :k]
        np.testing.assert_array_equal(actual_plan, cpu_plan.data[:, :k], err_msg=str((context, "signed plan")))
        if seeded_state and step == 0:
            np.testing.assert_array_equal(mapping.data[2 * physical : 2 * physical + rows], 0)
            # Reversed LRU visits slot 0 last among the duplicate token-3 hits.
            np.testing.assert_array_equal(actual_plan[:, 0], 1)
            np.testing.assert_array_equal(actual_plan[:, 1:], 0)
        np.testing.assert_array_equal(
            payload.data[:, wire],
            cpu_map.data[2 * physical : 2 * physical + rows],
            err_msg=str((context, "current token slots")),
        )
        for row in range(physical):
            bank = int(banks.data[row])
            rank_slots = state.data[row, bank, 1]
            rank_tokens = state.data[row, bank, 0]
            if seeded_state and step == 0 and row < rows:
                # Token 19 is speculative (stable=18); signed/out-of-domain
                # stale IDs retain their raw values under the CPU contract.
                for slot, token in ((2, -1), (3, -927), (4, maximum + 3)):
                    np.testing.assert_array_equal(rank_tokens[rank_slots == slot], [token])
            np.testing.assert_array_equal(rank_slots, cpu_state.data[1, row], err_msg=str((context, row, "LRU")))
            np.testing.assert_array_equal(
                rank_tokens, cpu_state.data[0, row, rank_slots], err_msg=str((context, row, "resident KV identity"))
            )
        # A plan's nonzero slot must identify the requested token after the
        # write phase. This contract check is separate from CPU result equality.
        for logical, physical_row in enumerate(mapping.data[:rows]):
            plan = actual_plan[logical].astype(np.int32)
            selected = plan != 0
            selected_slots = np.abs(plan[selected]) - 1
            np.testing.assert_array_equal(
                cpu_state.data[0, physical_row, selected_slots],
                topk.data[logical, selected],
                err_msg=str((context, logical, "plan KV identity")),
            )
        lib.publish(payload.ptr, membership.ptr, metadata.ptr, rows, k)
        np.testing.assert_array_equal(membership.data[:, : 16384 - k], 12345)
        np.testing.assert_array_equal(membership.data[:, 16384 - k : 16384], actual_plan)
        np.testing.assert_array_equal(metadata.data[1 : rows + 1], payload.data[:, wire])
        np.testing.assert_array_equal(metadata.data[rows + 1 :], 12345)
        for buffer in (
            state,
            cpu_state,
            last,
            cpu_last,
            banks,
            payload,
            mapping,
            membership,
            metadata,
            req,
            stable,
            visible,
            topk,
            cpu_map,
            cpu_plan,
        ):
            buffer.check()
        checks += rows
    return checks


def check_integer_keys(lib):
    # Integer ordering is the independent reference. No dense max-token table.
    values = np.array(
        [
            0,
            1,
            2,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 30) - 1,
            1 << 30,
            (1 << 30) + 1,
            (1 << 31) - 2,
            (1 << 31) - 1,
        ],
        dtype=np.int32,
    )
    keys = np.zeros(values.size, np.float32)
    scalar = np.zeros_like(keys)
    lib.token_keys(values.ctypes.data, keys.ctypes.data, scalar.ctypes.data, values.size)
    np.testing.assert_array_equal(keys.view(np.uint32), scalar.view(np.uint32))
    assert np.all(np.isfinite(keys)) and np.all(keys != 0)
    assert np.all(np.abs(keys) >= np.finfo(np.float32).tiny)
    assert np.all(np.diff(keys) > 0), (values, keys)
    source = np.resize(values, 64).copy()
    source[0] = -1
    source[1] = np.iinfo(np.int32).min
    sorted_keys, first = np.zeros(64, np.float32), np.zeros(64, np.float32)
    lib.sort_topk(source.ctypes.data, sorted_keys.ctypes.data, first.ctypes.data, source.size)
    assert np.all(np.diff(sorted_keys) <= 0)
    original_keys, ignored = np.zeros(64, np.float32), np.zeros(64, np.float32)
    lib.token_keys(source.ctypes.data, original_keys.ctypes.data, ignored.ctypes.data, source.size)
    for key, pos in zip(sorted_keys, first):
        assert int(pos) == int(np.flatnonzero(original_keys == key)[0]), (key, pos)
    return int(source.size + values.size)


def check_last_physical_row(lib):
    """Reach int16 row 32767 without 32768 active rows or quadratic reset work."""
    physical, rows, k, c = 32768, 3, 16, 16
    req = Guarded(rows, np.int64)
    req.data[:] = [999, 1000, 1001]
    last = Guarded(physical, np.int64, -1)
    last.data[-1] = 999
    cpu_last = Guarded(physical, np.int64)
    cpu_last.data[:] = last.data
    banks = Guarded(physical)
    state = Guarded((physical, 2, 2, c), fill=-1)
    state.data[:, :, 1] = np.arange(c)
    cpu_state = Guarded((2, physical, c), fill=-1)
    cpu_state.data[1] = np.arange(c)
    topk = Guarded((rows, k))
    topk.data[:] = np.arange(k)
    stable = Guarded(rows, fill=32)
    visible = Guarded(rows, fill=32)
    mapping = Guarded(4 * physical, fill=-777)
    cpu_map = Guarded(3 * physical, fill=-777)
    wire = ((k + 17) // 2 + 7) // 8 * 8
    payload = Guarded((rows, wire + 8))
    cpu_plan = Guarded((rows, k + 16), np.int16)
    lib.cpu_batch(
        req.ptr,
        cpu_last.ptr,
        topk.ptr,
        stable.ptr,
        visible.ptr,
        cpu_state.ptr,
        cpu_map.ptr,
        cpu_plan.ptr,
        rows,
        physical,
        k,
        c,
        64,
    )
    lib.plan_batch(
        req.ptr,
        topk.ptr,
        stable.ptr,
        visible.ptr,
        last.ptr,
        banks.ptr,
        state.ptr,
        mapping.ptr,
        payload.ptr,
        rows,
        physical,
        32,
        k,
        c,
        192 * 1024,
        64,
    )
    np.testing.assert_array_equal(mapping.data[:rows], [32767, 0, 1])
    np.testing.assert_array_equal(mapping.data[:rows], cpu_map.data[:rows])
    actual = payload.data[:, :wire].copy().view(np.int16).reshape(rows, 2 * wire)
    np.testing.assert_array_equal(actual[:, :k], cpu_plan.data[:, :k])
    np.testing.assert_array_equal(actual[:, k + 4], [32767, 0, 1])
    # C==K reserves the final current-token slot, so the last miss stays unassigned.
    np.testing.assert_array_equal(actual[:, k - 1], 0)
    for buffer in (
        req,
        last,
        cpu_last,
        banks,
        state,
        cpu_state,
        topk,
        stable,
        visible,
        mapping,
        cpu_map,
        payload,
        cpu_plan,
    ):
        buffer.check()
    return rows


def source_hashes(source):
    result = {}
    pending = [source]
    while pending:
        current = pending.pop()
        key = str(current.resolve())
        if key in result:
            continue
        data = current.read_bytes()
        result[key] = hashlib.sha256(data).hexdigest()
        for name in re.findall(r'^\s*#include\s+"([^"]+)"', data.decode("utf-8"), re.M):
            included = current.parent / name
            if included.is_file():
                pending.append(included)
    fixture = Path(__file__).with_name("fixtures") / "fsa_planner_host_shim.h"
    result[str(fixture.resolve())] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    return result


def run(args):
    source = args.source
    cpu_source = (
        args.cpu_source
        or source.parents[4] / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp"
    )
    lib = build(source, cpu_source, args.work_dir, args.cxx)
    dimensions = [
        (1, 2),
        (1, 16),
        (2, 2),
        (3, 3),
        (3, 5),
        (16, 16),
        (31, 33),
        (32, 64),
        (33, 65),
        (127, 129),
        (129, 257),
        (511, 769),
        (1023, 1025),
        (1537, 2049),
        (2047, 4095),
        (2048, 4096),
        (2048, 8193),
        (33, 32767),
    ]
    key_checks = check_integer_keys(lib)
    checks = 0
    for k, c in dimensions:
        for ub in (192 * 1024, 256 * 1024):
            checks += check_sequence(lib, k, c, 1, [1] * 8, 257, ub, 8, 72851 + k + c)
    for maximum in ((1 << 20) + 153, (1 << 24) + 257):
        for k, c in ((33, 65), (2048, 8193)):
            checks += check_sequence(lib, k, c, 1, [1] * 6, maximum, 192 * 1024, 8, 8791)
    for page in (8, 32, 104):
        checks += check_sequence(lib, 33, 65, 97, [33, 1, 48, 7, 97, 2, 31, 33], 257, 192 * 1024, page, 917)
    # Host-only larger UB forces an 8192-lane state sort across repeat=255.
    # This tests instruction chunking, not an available A3 hardware configuration.
    checks += check_sequence(lib, 33, 32767, 1, [1] * 4, 257, 384 * 1024, 8, 55231)
    checks += check_last_physical_row(lib)
    seeded_rows = 0
    for maximum in (257, (1 << 24) + 257):
        for k, c, ub in ((33, 65, 192 * 1024), (2048, 4096, 185664), (2048, 4096, 186688)):
            seeded_rows += check_sequence(
                lib, k, c, 3, [3, 1, 3, 2, 3, 1, 3, 1], maximum, ub, 8, 131, seeded_state=True
            )
    checks += seeded_rows
    result = {
        "passed_rows": checks,
        "seeded_state_rows": seeded_rows,
        "integer_key_checks": key_checks,
        "dimensions": dimensions,
        "oracle": "Unchanged production CPU lru_resident_compact_impl and process_one_lru_resident_row",
        "source_sha256": source_hashes(source),
        "cpu_source_sha256": hashlib.sha256(cpu_source.read_bytes()).hexdigest(),
        "coverage": [
            "non-power-of-two K/C",
            "capacity larger than UB",
            "token IDs above 2^24",
            "duplicate TopK/resident tokens/request IDs",
            "recurrent state without restoration",
            "matching pre-existing requests with duplicate resident tokens and reversed LRU",
            "seeded speculative invalidation and signed/out-of-domain stale tokens",
            "partial DMA",
            "physical rows >32",
            "graph-size buffer switching",
            "publish canaries",
        ],
        "not_verified": [
            "NPU instructions and event semantics",
            "actual graph capture/replay",
            "TP broadcasts and overlapping microbatches",
            "performance",
        ],
    }
    (args.work_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--cpu-source", type=Path)
    run(parser.parse_args())
