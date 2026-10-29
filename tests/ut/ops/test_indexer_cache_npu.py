# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B023
# Per-case callbacks run synchronously; checks read the current stage snapshots.
"""Explicit NPU whole-cache oracle gate for the generic row-scatter helpers.

No timing or service control. Torch is imported only after parsing the CLI.
Non-ND and invalid-slot probes are explicit. Known CANN output-layout rejections
retain their evidence and use a separately labelled logical reference check.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ORIGINAL_DTYPES = ("int8", "float16", "bfloat16", "float32", "int64", "bool")


def is_workspace_layout_rejection(error, cache_is_contiguous):
    return (
        isinstance(error, RuntimeError)
        and not cache_is_contiguous
        and "561103" in str(error)
        and "Output tensor should not from workspace" in str(error)
    )


def map_logical_cache_to_backing(before, logical, rows, width, strides, element_bytes):
    """Map a contiguous logical result into the original backing without touching holes."""
    if len(logical) != rows * width * element_bytes:
        raise ValueError("Logical cache byte size mismatch")
    result = bytearray(before)
    for row in range(rows):
        for column in range(width):
            src = (row * width + column) * element_bytes
            dst = (row * strides[0] + column * strides[1]) * element_bytes
            if dst < 0 or dst + element_bytes > len(result):
                raise ValueError("Logical cache view exceeds backing")
            result[dst : dst + element_bytes] = logical[src : src + element_bytes]
    return bytes(result)


def self_test():
    error = RuntimeError("aclnnScatterNdUpdate failed, error code is 561103: Output tensor should not from workspace")
    assert is_workspace_layout_rejection(error, False)
    assert not is_workspace_layout_rejection(error, True)
    assert not is_workspace_layout_rejection(RuntimeError("Triton compilation failed"), False)
    assert not is_workspace_layout_rejection(RuntimeError("561103: another parameter error"), False)
    before = bytes(range(40))
    logical = bytes([0x01, 0x7E, 0x00, 0x80, 0xFF, 0x7F, 0x00, 0x00])
    mapped = map_logical_cache_to_backing(before, logical, 2, 2, (7, 2), 2)
    expected = bytearray(before)
    for dst, src in ((0, 0), (4, 2), (14, 4), (18, 6)):
        expected[dst : dst + 2] = logical[src : src + 2]
    assert mapped == bytes(expected)
    assert map_logical_cache_to_backing(before, b"", 0, 2, (7, 2), 2) == before
    assert map_logical_cache_to_backing(before, b"", 2, 0, (7, 2), 2) == before
    for args in ((before, b"", 2, 2, (7, 2), 2), (b"", logical, 2, 2, (7, 2), 2)):
        try:
            map_logical_cache_to_backing(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed reference accepted")
    print("PASS: layout rejection classification; strided bit-copy reference, holes, empty and invalid extents")


def patterned_bytes(rows, width, dtype, salt=0):
    if dtype == "bool":
        return bytes((index + salt) & 1 for index in range(rows * width)), 1
    patterns = {
        "int8": (1, [0, 1, 0x7F, 0x80, 0xFF]),
        "float16": (2, [0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7D01, 0x7FFF, 1, 0x3555]),
        "bfloat16": (2, [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0x7FFF, 1, 0x3F80]),
        "float32": (4, [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001, 0x7FFFFFFF, 1, 0x3F800000]),
        "int64": (8, [0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF]),
        "float8_e4m3fn": (1, [0, 0x80, 0x7F, 0xFF, 1, 0x38]),
        "float8_e5m2": (1, [0, 0x80, 0x7C, 0xFC, 0x7D, 0x7F, 1, 0x3C]),
    }
    size, bits = patterns[dtype]
    data = b"".join(
        (bits[(row * width + col + salt) % len(bits)] ^ ((row * 17 + salt) & 0xFF)).to_bytes(size, "little")
        for row in range(rows)
        for col in range(width)
    )
    return data, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--kernel-path", type=Path, required=True)
    parser.add_argument("--expected-kernel-sha256", required=True)
    parser.add_argument("--expected-resources-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-format", type=int, default=2)
    parser.add_argument("--values-format", type=int, default=2)
    parser.add_argument("--slots-format", type=int, default=2)
    parser.add_argument("--dtype", choices=ORIGINAL_DTYPES + ("float8_e4m3fn", "float8_e5m2"), action="append")
    parser.add_argument("--invalid-slot-probe", choices=("negative", "upper", "above"))
    args = parser.parse_args()
    if args.device < 0 or sys.byteorder != "little":
        parser.error("Explicit nonnegative NPU and little-endian host required")
    dtypes = args.dtype or list(ORIGINAL_DTYPES)
    original_domain = (
        not args.invalid_slot_probe
        and all(fmt == 2 for fmt in (args.cache_format, args.values_format, args.slots_format))
        and all(dtype in ORIGINAL_DTYPES for dtype in dtypes)
    )
    source = args.kernel_path.resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.expected_kernel_sha256.lower():
        parser.error("Candidate source SHA mismatch before NPU import")
    resource_source = source.with_name("kernel_resources.py").resolve(strict=True)
    resource_digest = hashlib.sha256(resource_source.read_bytes()).hexdigest()
    if resource_digest != args.expected_resources_sha256.lower():
        parser.error("Adjacent kernel_resources.py SHA mismatch before NPU import")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidate_sha256": digest,
        "resource_sha256": resource_digest,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": [],
        "cases": [],
        "oracle_rejections": [],
        "timing_collected": False,
        "dtypes": dtypes,
        "validation_domain": "original_nd" if original_domain else "explicit_probe",
        "original_domain_pass": False,
        "scope": "Full backing-cache bytes, aliases, immutable inputs; eager and live-input graphs",
        "slot_policy": "Last valid input wins; invalid-slot probes must first agree with the current CANN oracle",
    }
    last, keep_alive = {}, []
    previous_internal_format = None

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu

        torch.npu.set_device(args.device)
        # Match the model process, including its Triton utility replacements.
        importlib.import_module("vllm_ascend.patch.worker")
        import triton

        report["triton_next_power_of_2"] = triton.next_power_of_2.__module__
        device = torch.device("npu", args.device)
        spec = importlib.util.spec_from_file_location("generic_indexer_cache_candidate", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Candidate source changed during import")
        resource_module = importlib.import_module(module.get_device_resources.__module__)
        imported_resource_source = Path(resource_module.__file__).resolve(strict=True)
        imported_resource_digest = hashlib.sha256(imported_resource_source.read_bytes()).hexdigest()
        report["resource_import"] = {
            "adjacent_path": str(resource_source),
            "imported_path": str(imported_resource_source),
            "imported_sha256": imported_resource_digest,
        }
        if (
            imported_resource_digest != resource_digest
            or hashlib.sha256(resource_source.read_bytes()).hexdigest() != resource_digest
        ):
            raise RuntimeError("Imported kernel_resources.py differs from the verified adjacent candidate")
        resources = module.get_device_resources(args.device)
        report["device_resources"] = resources._asdict()
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        if any(fmt != 2 for fmt in (args.cache_format, args.values_format, args.slots_format)):
            report["phase"] = "internal_format_setup"
            previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
            previous = previous.decode() if isinstance(previous, bytes) else previous
            if previous not in ("enable", "disable"):
                raise RuntimeError(f"Cannot safely restore ALLOW_INTERNAL_FORMAT: {previous!r}")
            previous_internal_format = previous
            report["internal_format"] = {"before": previous, "requested": "enable"}
            torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": "enable"})
            save()

        def raw(tensor):
            nd = tensor if torch_npu.get_npu_format(tensor) == 2 else torch_npu.npu_format_cast(tensor, 2)
            bit_type = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[nd.element_size()]
            return nd.detach().view(bit_type).cpu().numpy().tobytes(order="C")

        def from_bytes(data, dtype, shape):
            return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().view(dtype).reshape(shape)

        def on_device(tensor, fmt):
            result = tensor.to(device)
            return result if fmt == 2 else torch_npu.npu_format_cast(result, fmt)

        # Exercise tails, physical strides and repeated destinations, not eligibility policies.
        layouts = [
            (0, 128, 1, 1, 1),
            (5, 0, 1, 1, 1),
            (1, 1, 1, 1, 1),
            (5, 3, 1, 1, 1),
            (6, 128, 1, 1, 1),
            (7, 31, 2, 2, 2),
            (17, 257, 2, 3, 2),
            (129, 1, 1, 2, 3),
            (257, 3, 2, 2, 2),
        ]
        layouts = [(*layout, False, False, "view") for layout in layouts] + [
            (7, 31, 1, 1, 1, True, False, "view"),
            (7, 31, 1, 1, 1, False, True, "view"),
            # Source-only views retain a contiguous output, so the actual
            # CANN interface must accept the same layout (no reference fallback).
            (6, 128, 1, 2, 1, False, False, "source_only"),
            (7, 33, 1, 17, 2, False, False, "source_only"),
            (17, 129, 1, 1, 1, False, False, "transpose"),
            (128, 17, 1, 1, 1, False, False, "transpose"),
            (6, 128, 1, 1, 1, False, False, "transpose"),
            (33, 129, 1, 1, 1, False, False, "transpose_gap"),
            (7, 31, 1, 1, 1, False, False, "column_zero"),
        ]
        if args.invalid_slot_probe:
            layouts = [(7, 3, 1, 1, 1, False, False, "view")]  # Run a risky native probe in its own process.
        with torch.inference_mode():
            for dtype_name in dtypes:
                dtype = getattr(torch, dtype_name)
                dtype_layouts = list(layouts)
                if original_domain and dtype_name == "int8":
                    # Derive boundary cases from this device rather than reproducing a fixed tuning block.
                    _, index_limit, _ = module._scatter_tiling(
                        resources.ub_bytes,
                        1,
                        resources.ub_bytes,
                        1,
                        8,
                        1,
                        resources.vector_cores,
                        resources.ub_bytes,
                    )
                    _, _, copy_limit = module._scatter_tiling(
                        1,
                        resources.ub_bytes,
                        17,
                        1,
                        8,
                        1,
                        resources.vector_cores,
                        resources.ub_bytes,
                    )
                    dtype_layouts += [
                        (max(1, resources.vector_cores - 1), 3, 1, 1, 1, False, False, "view"),
                        (resources.vector_cores + 1, 3, 1, 1, 1, False, False, "view"),
                        (index_limit + 1, 1, 1, 1, 1, False, False, "view"),
                        (3, copy_limit + 1, 1, 1, 1, False, False, "view"),
                    ]
                for layout_index, layout in enumerate(dtype_layouts):
                    n, width, row_step, col_step, slot_step, value_row_zero, slot_zero, value_layout = layout
                    capacity = max(n + 3, 17)
                    cache_row_step = row_step if value_layout == "view" else 1
                    cache_col_step = col_step if value_layout == "view" else 1
                    base_shape = (
                        (capacity * row_step + 1, width * col_step + 3) if value_layout == "view" else (capacity, width)
                    )
                    value_shape = (max(n * row_step, 1), width * col_step + 3)
                    if value_layout in ("transpose", "transpose_gap"):
                        value_shape = (width, n + (5 if value_layout == "transpose_gap" else 0))
                    elif value_layout == "column_zero":
                        value_shape = (n, 1)
                    payload, size = patterned_bytes(*value_shape, dtype_name)
                    initial_byte = 0 if dtype_name == "bool" else 0x5A
                    initial = from_bytes(
                        bytes([initial_byte]) * (base_shape[0] * base_shape[1] * size), dtype, base_shape
                    )
                    value_cpu = from_bytes(payload, dtype, value_shape)
                    changed_payload, _ = patterned_bytes(*value_shape, dtype_name, salt=13)
                    changed_cpu = from_bytes(changed_payload, dtype, value_shape)
                    cpu_cache_view = initial[
                        : capacity * cache_row_step : cache_row_step, : width * cache_col_step : cache_col_step
                    ]
                    patterns = (
                        ("unique",) if not n or not width else ("unique", "padding", "duplicate_pairs", "duplicate_all")
                    )
                    if slot_zero:
                        patterns = ("padding", "duplicate_all")
                    for pattern in patterns:
                        label = f"{dtype_name}/n{n}/w{width}/r{row_step}/c{col_step}/s{slot_step}/{pattern}"
                        if value_layout != "view":
                            label += "/" + value_layout
                        if value_row_zero:
                            label += "/values_row_stride0"
                        if slot_zero:
                            label += "/slots_stride0"
                        last.clear()
                        last["case"] = label
                        report["phase"] = "prepare/" + label
                        save()
                        slots_list = [index for index in range(n)]
                        if n:
                            slots_list[-1] = capacity - 1
                        if pattern == "padding":
                            slots_list = [-1 if index % 2 else value for index, value in enumerate(slots_list)]
                        elif pattern == "duplicate_pairs":
                            slots_list = [index // 2 for index in range(n)]
                        elif pattern == "duplicate_all":
                            slots_list = [capacity - 1] * n
                        if args.invalid_slot_probe:
                            slots_list[0] = {"negative": -2, "upper": capacity, "above": capacity + 7}[
                                args.invalid_slot_probe
                            ]
                        if slot_zero:
                            slots_list = [-1 if pattern == "padding" else capacity - 1] * n
                        slot_dtype = torch.int64 if layout_index % 2 else torch.int32
                        slot_cpu = torch.full((max(n * slot_step, 1),), -1, dtype=slot_dtype)
                        slot_cpu[: n * slot_step : slot_step] = torch.tensor(slots_list, dtype=slot_dtype)
                        values_base = on_device(value_cpu, args.values_format)
                        slots_base = on_device(slot_cpu, args.slots_format)
                        if value_layout in ("transpose", "transpose_gap"):
                            values = values_base[:, :n].transpose(0, 1)
                        elif value_layout == "column_zero":
                            values = values_base.expand(n, width)
                        else:
                            values = values_base[: n * row_step : row_step, : width * col_step : col_step]
                        slots = slots_base[: n * slot_step : slot_step].view(n, 1)
                        if value_row_zero:
                            values = values[:1].expand(n, width)
                        if slot_zero:
                            slots = slots[:1].expand(n, 1)
                        bases = {mode: on_device(initial, args.cache_format) for mode in ("cann", "triton")}
                        views = {
                            mode: base[
                                : capacity * cache_row_step : cache_row_step, : width * cache_col_step : cache_col_step
                            ]
                            for mode, base in bases.items()
                        }
                        initial_bits = initial.view(torch.uint8).numpy().tobytes()
                        if any(raw(base) != initial_bits for base in bases.values()):
                            raise RuntimeError("Cache H2D/format conversion changed raw bits")
                        function = module.scatter_k_cache if layout_index % 2 == 0 else module.scatter_scale_cache
                        case = {
                            "case": label,
                            "function": function.__name__,
                            "slots": slots_list,
                            "oracle_scope": "same_layout_cann",
                            "metadata": {
                                name: {
                                    "shape": list(tensor.shape),
                                    "stride": list(tensor.stride()),
                                    "storage_offset": tensor.storage_offset(),
                                    "dtype": str(tensor.dtype),
                                }
                                for name, tensor in (("cache", views["triton"]), ("values", values), ("slots", slots))
                            },
                            "formats": {
                                "cache": torch_npu.get_npu_format(views["triton"]),
                                "values": torch_npu.get_npu_format(values),
                                "slots": torch_npu.get_npu_format(slots),
                            },
                        }
                        if n and width:
                            plan = module._scatter_tiling(
                                n,
                                width,
                                capacity,
                                size,
                                slots.element_size(),
                                values.stride(1),
                                resources.vector_cores,
                                resources.ub_bytes,
                            )
                            case["tiling"] = dict(zip(("owners", "index_block", "copy_block"), plan))
                        report["cases"].append(case)
                        for name, tensor, fmt in (
                            ("cache", views["triton"], args.cache_format),
                            ("values", values, args.values_format),
                            ("slots", slots, args.slots_format),
                        ):
                            if torch_npu.get_npu_format(tensor) != fmt:
                                raise RuntimeError(f"Requested physical format was not preserved for {name}")
                        if (
                            raw(values_base) != payload
                            or raw(slots_base) != slot_cpu.view(torch.uint8).numpy().tobytes()
                        ):
                            raise RuntimeError("Source/slot H2D changed raw bits")
                        graphs, graph_outputs = {}, {}

                        def invoke(mode, cache, operation="eager"):
                            report["phase"] = f"{mode}/{operation}/{label}/{stage}"
                            return (
                                torch_npu.npu_scatter_nd_update_(cache, slots, values)
                                if mode == "cann"
                                else function(cache, slots, values)
                            )

                        def compare(mode, stage, actual, expected, input_bits, slot_bits):
                            report["phase"] = f"{mode}/compare/{label}/{stage}"
                            blob = raw(bases[mode])
                            actual_values, actual_slots = raw(values_base), raw(slots_base)
                            last.update(
                                actual=blob,
                                expected=expected,
                                values=input_bits,
                                slots_raw=slot_bits,
                                values_backing_before=value_backing_bits,
                                slots_backing_before=slot_backing_bits,
                                values_backing_actual=actual_values,
                                slots_backing_actual=actual_slots,
                            )
                            passed = (
                                blob == expected
                                and actual.data_ptr() == views[mode].data_ptr()
                                and actual.shape == views[mode].shape
                                and actual.dtype == views[mode].dtype
                                and actual.device == views[mode].device
                                and raw(values) == input_bits
                                and raw(slots) == slot_bits
                                and actual_values == value_backing_bits
                                and actual_slots == slot_backing_bits
                            )
                            report["checks"].append(
                                {
                                    "case": label,
                                    "mode": mode,
                                    "stage": stage,
                                    "oracle_scope": case["oracle_scope"],
                                    "pass": passed,
                                    "actual_sha256": hashlib.sha256(blob).hexdigest(),
                                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                                }
                            )
                            if not passed:
                                raise RuntimeError(f"Whole-cache/alias/input gate failed: {label}/{mode}/{stage}")

                        for stage in ("original", "changed", "restored"):
                            values_base.copy_(on_device(changed_cpu if stage == "changed" else value_cpu, 2))
                            if raw(values_base) != (changed_payload if stage == "changed" else payload):
                                raise RuntimeError("Live source H2D/copy changed raw bits")
                            stage_slots = (
                                [value if value < 0 else (value + 3) % capacity for value in slots_list]
                                if stage == "changed"
                                else slots_list
                            )
                            stage_slots_cpu = torch.tensor(stage_slots, dtype=slot_dtype).reshape(n, 1)
                            if slot_zero:
                                slots_base[0].copy_(stage_slots_cpu[0, 0].to(device))
                            else:
                                slots.copy_(stage_slots_cpu.to(device))
                            input_bits, slot_bits = raw(values), raw(slots)
                            value_backing_bits, slot_backing_bits = raw(values_base), raw(slots_base)
                            if slot_bits != stage_slots_cpu.view(torch.uint8).numpy().tobytes():
                                raise RuntimeError("Live slot H2D/copy changed raw bits")
                            oracle_base = on_device(initial, args.cache_format)
                            oracle_view = oracle_base[
                                : capacity * cache_row_step : cache_row_step, : width * cache_col_step : cache_col_step
                            ]
                            last.update(before=raw(oracle_base), values=input_bits, slots_raw=slot_bits)
                            if last["before"] != initial_bits:
                                raise RuntimeError("Oracle cache initialization changed raw bits")
                            report["phase"] = f"cann_oracle/{label}/{stage}"
                            save()
                            if case["oracle_scope"] == "same_layout_cann":
                                try:
                                    invoke("cann", oracle_view, "oracle")
                                except RuntimeError as error:
                                    if not is_workspace_layout_rejection(error, oracle_view.is_contiguous()):
                                        raise
                                    rejection_dir = output / "oracle-rejections" / str(len(report["oracle_rejections"]))
                                    rejection_dir.mkdir(parents=True)
                                    evidence = {
                                        "cache_before": last["before"],
                                        "cache_after_rejection": raw(oracle_base),
                                        "values": input_bits,
                                        "slots": slot_bits,
                                        "values_backing": value_backing_bits,
                                        "slots_backing": slot_backing_bits,
                                    }
                                    for name, blob in evidence.items():
                                        (rejection_dir / f"{name}.bin").write_bytes(blob)
                                    report["oracle_rejections"].append(
                                        {
                                            "case": label,
                                            "stage": stage,
                                            "phase": report["phase"],
                                            "status": "oracle-rejected",
                                            "error": repr(error),
                                            "traceback": traceback.format_exc(),
                                            "metadata": case["metadata"],
                                            "raw_directory": str(rejection_dir.relative_to(output)),
                                            "raw_sha256": {
                                                name: hashlib.sha256(blob).hexdigest()
                                                for name, blob in evidence.items()
                                            },
                                        }
                                    )
                                    case["oracle_scope"] = "contiguous_cann_reference_only"
                                    save()
                            if case["oracle_scope"] == "same_layout_cann":
                                expected = raw(oracle_base)
                            else:
                                report["phase"] = f"cann/contiguous_reference/{label}/{stage}"
                                reference_cache = on_device(
                                    from_bytes(
                                        bytes([initial_byte]) * (capacity * width * size), dtype, (capacity, width)
                                    ),
                                    2,
                                )
                                reference_values = on_device(from_bytes(input_bits, dtype, (n, width)), 2)
                                reference_slots = on_device(from_bytes(slot_bits, slot_dtype, (n, 1)), 2)
                                reference_result = torch_npu.npu_scatter_nd_update_(
                                    reference_cache, reference_slots, reference_values
                                )
                                if (
                                    reference_result.data_ptr() != reference_cache.data_ptr()
                                    or reference_result.shape != reference_cache.shape
                                    or reference_result.dtype != reference_cache.dtype
                                    or reference_result.device != reference_cache.device
                                    or raw(reference_values) != input_bits
                                    or raw(reference_slots) != slot_bits
                                ):
                                    raise RuntimeError("Contiguous CANN reference changed its alias or inputs")
                                expected = map_logical_cache_to_backing(
                                    initial_bits, raw(reference_cache), capacity, width, cpu_cache_view.stride(), size
                                )
                            last["expected"] = expected
                            expected_sha = hashlib.sha256(expected).hexdigest()
                            if stage == "original":
                                case["original_oracle_sha256"] = expected_sha
                            elif (
                                stage == "changed" and n and width and any(0 <= slot < capacity for slot in stage_slots)
                            ):
                                if expected_sha == case["original_oracle_sha256"]:
                                    raise RuntimeError("Changed inputs did not change the current CANN oracle")
                            elif stage == "restored" and expected_sha != case["original_oracle_sha256"]:
                                raise RuntimeError("Restored inputs did not restore the CANN oracle")
                            golden = bytearray(initial.view(torch.uint8).numpy().tobytes())
                            for index, slot in enumerate(stage_slots):
                                if 0 <= slot < capacity:
                                    for column in range(width):
                                        dst = (
                                            slot * cpu_cache_view.stride(0) + column * cpu_cache_view.stride(1)
                                        ) * size
                                        src = (index * width + column) * size
                                        golden[dst : dst + size] = input_bits[src : src + size]
                            if expected != golden:
                                last["last_wins_expected"] = bytes(golden)
                                raise RuntimeError(
                                    "Current CANN disagrees with the proposed last-wins/ignored-invalid policy"
                                )
                            modes = ("cann", "triton") if case["oracle_scope"] == "same_layout_cann" else ("triton",)
                            for mode in modes:
                                bases[mode].copy_(on_device(initial, 2))
                                compare(
                                    mode, stage + "/eager", invoke(mode, views[mode]), expected, input_bits, slot_bits
                                )
                            if not n or not width:
                                continue
                            if not graphs:
                                for mode in modes:
                                    report["phase"] = f"{mode}/capture/{label}/{stage}"
                                    graph = torch.npu.NPUGraph()
                                    with torch.npu.graph(graph):
                                        graph_outputs[mode] = invoke(mode, views[mode], "capture")
                                    graphs[mode] = graph
                                keep_alive.append(
                                    (graphs, graph_outputs, bases, views, values_base, slots_base, values, slots)
                                )
                            for repeat in range(2):
                                for mode in modes:
                                    report["phase"] = f"{mode}/replay_reset/{label}/{stage}/{repeat}"
                                    bases[mode].copy_(on_device(initial, 2))
                                    report["phase"] = f"{mode}/replay/{label}/{stage}/{repeat}"
                                    graphs[mode].replay()
                                    torch.npu.synchronize()
                                    compare(
                                        mode,
                                        f"{stage}/graph{repeat}",
                                        graph_outputs[mode],
                                        expected,
                                        input_bits,
                                        slot_bits,
                                    )
                            save()
        if not report["checks"] or not all(check["pass"] for check in report["checks"]):
            raise RuntimeError("No complete passing checks")
        reference_cases = sum(case["oracle_scope"] != "same_layout_cann" for case in report["cases"])
        report.update(
            status=("passed_with_reference_only" if reference_cases else "passed")
            if original_domain
            else "probe_passed",
            original_domain_pass=original_domain and not reference_cases,
            oracle_scope_counts={
                "same_layout_cann": len(report["cases"]) - reference_cases,
                "reference_only": reference_cases,
            },
            phase="complete",
            retained_graph_count=sum(len(item[0]) for item in keep_alive),
        )
        save()
    except Exception as error:
        failure = output / "first-failure"
        failure.mkdir(exist_ok=True)
        for name, value in last.items():
            if isinstance(value, bytes):
                (failure / f"{name}.bin").write_bytes(value)
        report.update(
            status="failed", error=repr(error), traceback=traceback.format_exc(), failure_case=last.get("case")
        )
        save()
        raise
    finally:
        if previous_internal_format is not None:
            try:
                torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous_internal_format})
                restored = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
                restored = restored.decode() if isinstance(restored, bytes) else restored
                report["internal_format"].update(after=restored, restored=restored == previous_internal_format)
                if restored != previous_internal_format:
                    raise RuntimeError(f"ALLOW_INTERNAL_FORMAT restoration failed: {report['internal_format']}")
            except Exception as error:
                report.update(
                    status="failed",
                    restore_error=repr(error),
                    failure_phase_before_restore=report.get("phase"),
                    phase="internal_format_restore",
                )
                save()
                raise
            save()


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
