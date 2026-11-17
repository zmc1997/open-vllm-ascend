# SPDX-License-Identifier: Apache-2.0
"""Opt-in NPU planner graph gate. Never executes without --run-device.

Compare alternating graph sizes against eager calls on separate recurrent state.
This is a graph/address-lifetime gate, not an independent numerical oracle; run
fsa_planner_geometry_host.py separately for the original CPU planner comparison.
No model is loaded and no formal Bench is run. Not exercised by CPU CI.
"""

import argparse
import ctypes


def run(library_path, device_index):
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.kernel_resources import get_device_resources

    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    resources = get_device_resources(device.index)
    cores, ub = resources.vector_cores, resources.ub_bytes
    library = ctypes.CDLL(library_path)
    plan = library.fsa_lru_plan_async_v5
    plan.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 7 + [ctypes.c_int32, ctypes.c_void_p]
    plan.restype = ctypes.c_int
    publish = library.fsa_lru_publish_async_v4
    publish.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    publish.restype = ctypes.c_int

    for k, c, physical in ((16, 16, 7), (33, 128, 33), (2048, 8192, 7)):
        for maximum in ((1 << 20) + 153, (1 << 24) + 257):
            wire = ((k + 17) // 2 + 7) // 8 * 8
            page = min((physical + 7) // 8 * 8, ub // 36 // 8 * 8)

            def persistent(physical=physical, c=c):
                state = torch.full((physical, 2, 2, c), -1, dtype=torch.int32, device=device)
                state[:, :, 1] = torch.arange(c, dtype=torch.int32, device=device)
                return (
                    torch.full((physical,), -1, dtype=torch.int64, device=device),
                    torch.zeros(physical, dtype=torch.int32, device=device),
                    state,
                )

            def buffers(rows, k=k, maximum=maximum, physical=physical, wire=wire):
                return (
                    torch.arange(100, 100 + rows, dtype=torch.int64, device=device),
                    torch.full((rows, k), -1, dtype=torch.int32, device=device),
                    torch.full((rows,), maximum - 2, dtype=torch.int32, device=device),
                    torch.full((rows,), maximum - 2, dtype=torch.int32, device=device),
                    torch.empty(4 * physical, dtype=torch.int32, device=device),
                    torch.zeros((rows, wire + 8), dtype=torch.int32, device=device),
                    torch.full((rows, 16400), -117, dtype=torch.int16, device=device),
                    torch.full((rows + 1,), -117, dtype=torch.int32, device=device),
                )

            def execute(buf, state, physical=physical, page=page, k=k, c=c, maximum=maximum):
                req, topk, stable, visible, mapping, payload, members, metadata = buf
                last, banks, recurrent = state
                rows = req.numel()
                # Resolve inside the graph context so the kernels enter its capture stream.
                stream = torch.npu.current_stream(device)
                ptrs = [t.data_ptr() for t in (req, topk, stable, visible, last, banks, recurrent, mapping, payload)]
                status = plan(*ptrs, rows, physical, cores, page, k, c, ub, maximum, stream.npu_stream)
                assert status == 0, ("plan", status)
                status = publish(
                    payload.data_ptr(), members.data_ptr(), metadata.data_ptr(), rows, cores, k, stream.npu_stream
                )
                assert status == 0, ("publish", status)

            captured_state, eager_state = persistent(), persistent()
            graphs, graph_buffers, eager_buffers = {}, {}, {}
            sizes = (1, 3, physical)
            for rows in sizes:
                graph_buffers[rows], eager_buffers[rows] = buffers(rows), buffers(rows)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    execute(graph_buffers[rows], captured_state)
                graphs[rows] = graph
            # Capture may execute operations. Reset once before the recurrent sequence,
            # then never restore initial state between graph switches or hit reads.
            for dst, src in zip(captured_state, eager_state):
                dst.copy_(src)
            torch.npu.synchronize()
            for step, rows in enumerate((1, physical, 3, 1, 3, physical, 1)):
                actual, reference = graph_buffers[rows], eager_buffers[rows]
                ids = torch.arange(100, 100 + rows, dtype=torch.int64, device=device)
                if step % 2:
                    ids = ids.flip(0)
                if step == 4:
                    ids[0] = 999
                actual[0].copy_(ids)
                visible = maximum - 2 + step % 2
                actual[1].fill_(-1)
                values = [visible - 2, visible - 1, visible, visible - 2, -1, 3]
                actual[1][:, : len(values)] = torch.tensor(values, dtype=torch.int32, device=device)
                actual[2].fill_(visible - step % 3)
                actual[3].fill_(visible)
                for dst, src in zip(reference[:4], actual[:4]):
                    dst.copy_(src)
                graphs[rows].replay()
                execute(reference, eager_state)
                torch.npu.synchronize()
                for name, lhs, rhs in zip(("payload", "membership", "metadata"), actual[5:], reference[5:]):
                    assert torch.equal(lhs, rhs), (k, c, physical, maximum, step, name)
                for name, lhs, rhs in zip(("last IDs", "banks", "state"), captured_state, eager_state):
                    assert torch.equal(lhs, rhs), (k, c, physical, maximum, step, name)
            print(f"PASS graph switches/recurrent state K={k} C={c} rows={physical} max={maximum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--run-device", action="store_true", help="explicitly enable device execution")
    args = parser.parse_args()
    if not args.run_device:
        parser.error("NPU execution requires --run-device; this gate is not part of CPU-only validation")
    run(args.library, args.device)
