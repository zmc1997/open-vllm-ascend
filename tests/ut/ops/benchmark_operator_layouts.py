# SPDX-License-Identifier: Apache-2.0
"""Isolated, byte-gated full-call comparison across shapes and layouts.

Copies made inside either public operator are included. Graph-event intervals
are amortized over repeats; these are not model ITL or individual kernel times.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import statistics
import sys
import traceback
from functools import partial
from pathlib import Path


def load(name, path, resource_name=None, registered_module=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if registered_module is not None:
        # The two frozen BMM versions share identical Cube control operations.
        # Reuse their registration in this isolated comparison process only;
        # do not mutate the backend registry or double-register the same name.
        original = ast.parse(Path(registered_module.__file__).read_text(encoding="utf-8"))
        classes = {node.name: node for node in original.body if isinstance(node, ast.ClassDef)}
        for node in list(tree.body):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(d, ast.Attribute) and d.attr == "register_custom_op" for d in node.decorator_list
            ):
                assert ast.dump(node) == ast.dump(classes[node.name]), "Custom-op definitions differ"
                operation = getattr(registered_module, node.name)
                assert path.with_name(Path(operation.bitcode).name).read_bytes() == Path(operation.bitcode).read_bytes()
                module.__dict__[node.name] = operation
                tree.body.remove(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.ops.triton.kernel_resources":
            node.module = resource_name
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def run(args, report):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.manual_seed(20261029)
    device = f"npu:{args.device}"
    resource = load("_layout_resources", args.source_dir / "kernel_resources.py")
    modules = {}
    for name in ("static_quantize", "vup_bmm", "indexer_cache"):
        if {"static_quantize": "quant", "vup_bmm": "bmm", "indexer_cache": "cache"}[name] in args.kind:
            path = args.source_dir / (name + ".py")
            modules[name] = load("_layout_" + name, path, "_layout_resources")
            report["sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.previous_quant:
        previous = load("_layout_previous", args.previous_quant, "_layout_resources")
        report["sources"]["previous_quant"] = hashlib.sha256(args.previous_quant.read_bytes()).hexdigest()
    previous_modules = {}
    if args.previous_dir:
        load("_layout_previous_resources", args.previous_dir / "kernel_resources.py")
        for name in modules:
            path = args.previous_dir / (name + ".py")
            previous_modules[name] = load(
                "_layout_previous_" + name,
                path,
                "_layout_previous_resources",
                modules[name] if name == "vup_bmm" else None,
            )
            report["sources"]["previous_" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["resources"] = resource.get_device_resources(args.device)._asdict()
    report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}

    def bits(t):
        return (
            t.contiguous()
            .view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])
            .cpu()
            .numpy()
            .tobytes()
        )

    def tensor(shape, dtype, layout):
        if layout == "transpose":
            return (
                (torch.randn(tuple(reversed(shape))) * 0.1)
                .to(device=device, dtype=dtype)
                .permute(tuple(reversed(range(len(shape)))))
            )
        step = int(layout[6:]) if layout.startswith("stride") else 1
        value = (torch.randn((*shape[:-1], shape[-1] * step)) * 4).to(device=device, dtype=dtype)
        return value[..., ::step]

    def measure(metadata, calls, inputs, outputs=None):
        case = {**metadata, "status": "running", "checks": [], "samples": []}
        report["cases"].append(case)

        def save():
            args.output.write_text(json.dumps(report, indent=2))

        originals = [bits(t) for t in inputs]
        expected = calls["cann"]()
        expected_bits = bits(expected)

        def check(t, phase):
            assert t.dtype == expected.dtype and t.shape == expected.shape, phase
            assert bits(t) == expected_bits, phase
            if outputs is not None:
                assert t.data_ptr() == outputs[phase.split("/")[0]].data_ptr(), "cache alias changed"
            else:
                assert t.is_contiguous(), "output layout changed"
            case["checks"].append(phase)

        for mode, call in calls.items():
            check(call(), mode + "/eager")
        graphs, results = {}, {}
        for mode, call in calls.items():
            for _ in range(3):
                call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.repeats):
                    output = call()
            graph.replay()
            torch.npu.synchronize()
            check(output, mode + "/graph")
            graphs[mode], results[mode] = graph, output
        # Prove graph reads actual input storage. Restore before timing.
        first = inputs[0]
        saved = first.clone()
        first.zero_()
        expected = calls["cann"]()
        expected_bits = bits(expected)
        for mode, graph in graphs.items():
            graph.replay()
            torch.npu.synchronize()
            check(results[mode], mode + "/input_update")
        first.copy_(saved)
        expected = calls["cann"]()
        expected_bits = bits(expected)
        for _ in range(5):
            for graph in graphs.values():
                graph.replay()
        torch.npu.synchronize()
        for sample in range(args.samples):
            order = list(calls) if sample % 2 == 0 else list(reversed(calls))
            for mode in order:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                case["samples"].append(
                    {"sample": sample, "mode": mode, "us": start.elapsed_time(end) * 1000 / args.repeats}
                )
        for mode in calls:
            check(results[mode], mode + "/after_timing")
        assert [bits(t) for t in inputs] == originals, "input changed"
        case["inputs_unchanged"] = True
        case["mean_us"] = {
            mode: statistics.mean(s["us"] for s in case["samples"] if s["mode"] == mode) for mode in calls
        }
        case["status"] = "passed"
        save()
        print(json.dumps({k: case[k] for k in ("kind", "shape", "layout", "dtype", "mean_us")}), flush=True)

    for dtype_name in args.dtype:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
        if "quant" in args.kind:
            cases = [
                (6, 8192, "contiguous"),
                (6, 2048, "contiguous"),
                (128, 8192, "contiguous"),
                (512, 8192, "contiguous"),
                (513, 17, "contiguous"),
                (3, 257, "contiguous"),
                (6, 8193, "contiguous"),
                (6, 8192, "stride2"),
                (6, 2048, "stride2"),
                (128, 8192, "stride2"),
                (17, 257, "stride3"),
                (17, 129, "stride16"),
                (6, 2048, "transpose"),
                (128, 8192, "transpose"),
            ]
            for n, width, layout in cases:
                if args.layout and layout not in args.layout:
                    continue
                x = tensor((n, width), dtype, layout)
                scale = torch.tensor([0.5, 0.75, 1.0, 1.5], dtype=dtype, device=device).repeat((width + 3) // 4)[:width]
                offset = torch.tensor([-0.5, -0.25, 0.25, 0.5], dtype=dtype, device=device).repeat((width + 3) // 4)[
                    :width
                ]
                calls = {"cann": partial(torch_npu.npu_quantize, x, scale, offset, torch.qint8, -1, False)}
                if args.previous_quant:
                    calls["previous"] = partial(previous.quantize, x, scale, offset)
                elif "static_quantize" in previous_modules:
                    calls["previous"] = partial(previous_modules["static_quantize"].quantize, x, scale, offset)
                calls["current"] = partial(modules["static_quantize"].quantize, x, scale, offset)
                measure(
                    {
                        "kind": "quant",
                        "shape": [n, width],
                        "layout": layout,
                        "dtype": dtype_name,
                        "strides": list(x.stride()),
                    },
                    calls,
                    [x, scale, offset],
                )
        if "bmm" in args.kind:
            cases = [(1, 32, 512, 256), (6, 32, 512, 256), (128, 32, 512, 256), (7, 3, 33, 65), (33, 3, 513, 129)]
            for n, heads, k, v in cases:
                for layout in ("contiguous", "stride2", "transpose"):
                    if args.layout and layout not in args.layout:
                        continue
                    x, w = tensor((n, heads, k), dtype, layout), tensor((heads, k, v), dtype, layout)
                    calls = {
                        "cann": partial(torch_npu.npu_transpose_batchmatmul, x, w, perm_x1=(1, 0, 2), perm_y=(1, 0, 2)),
                        "current": partial(modules["vup_bmm"].vup_bmm, x, w),
                    }
                    if "vup_bmm" in previous_modules:
                        calls["previous"] = partial(previous_modules["vup_bmm"].vup_bmm, x, w)
                    measure(
                        {
                            "kind": "bmm",
                            "shape": [n, heads, k, v],
                            "layout": layout,
                            "dtype": dtype_name,
                            "input_strides": list(x.stride()),
                            "weight_strides": list(w.stride()),
                        },
                        calls,
                        [x, w],
                    )
    if "cache" in args.kind:
        for width, dtype, kind in (
            (128, torch.int8, "cache_k"),
            (17, torch.int8, "cache_k"),
            (256, torch.int8, "cache_k"),
            (1, torch.float16, "cache_scale"),
            (3, torch.float16, "cache_scale"),
        ):
            for n, distribution in ((6, "clustered"), (128, "clustered"), (128, "dispersed"), (128, "duplicates")):
                for layout in ("contiguous", "stride2", "transpose"):
                    if args.layout and layout not in args.layout:
                        continue
                    capacity = 32768
                    values = tensor((n, width), dtype, layout)
                    ids = torch.arange(n, dtype=torch.int32)
                    if distribution == "dispersed":
                        ids = ids * (capacity // n)
                    elif distribution == "duplicates":
                        ids = ids % 7
                    slots = ids.to(device).view(-1, 1)
                    a = torch.zeros((capacity, width), device=device, dtype=dtype)
                    b = torch.zeros_like(a)
                    fn = (
                        modules["indexer_cache"].scatter_k_cache
                        if kind == "cache_k"
                        else modules["indexer_cache"].scatter_scale_cache
                    )
                    calls = {
                        "cann": partial(torch_npu.npu_scatter_nd_update_, a, slots, values),
                        "current": partial(fn, b, slots, values),
                    }
                    outputs = {"cann": a, "current": b}
                    if "indexer_cache" in previous_modules:
                        previous_cache = torch.zeros_like(a)
                        name = "scatter_k_cache" if kind == "cache_k" else "scatter_scale_cache"
                        calls["previous"] = partial(
                            getattr(previous_modules["indexer_cache"], name), previous_cache, slots, values
                        )
                        outputs["previous"] = previous_cache
                    measure(
                        {
                            "kind": kind,
                            "shape": [n, width],
                            "layout": layout,
                            "dtype": str(dtype),
                            "slots": distribution,
                            "cache_shape": [capacity, width],
                            "strides": list(values.stride()),
                        },
                        calls,
                        [values, slots],
                        outputs,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--previous-quant", type=Path)
    parser.add_argument("--previous-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kind", choices=("quant", "bmm", "cache"), action="append", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), action="append", default=None)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--layout", action="append")
    args = parser.parse_args()
    args.dtype = args.dtype or ["bf16"]
    assert not (args.previous_quant and args.previous_dir), "Choose one frozen comparator source"
    assert args.repeats > 0 and args.samples > 0 and not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "sources": {},
        "cases": [],
        "repeats": args.repeats,
        "samples": args.samples,
        "layout_filter": args.layout,
        "scope": "full public operator, including copies; repeated synthetic input; graph-event us/call; no model",
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))
