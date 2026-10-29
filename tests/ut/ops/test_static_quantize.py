# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict NPU CANN oracle gate; no timing or model/service operations.

BF16, FP16 and FP32 ND candidates are gated independently on the actual runtime.
Direct NZ calls are explicit diagnostics outside the original QINT8 contract.
FP32 patterns are sampled, not an exhaustive enumeration of all 32-bit values.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu


def raw(tensor):
    if tensor.device.type == "npu" and torch_npu.get_npu_format(tensor) != 2:
        tensor = torch_npu.npu_format_cast(tensor, 2)
    bit_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    return tensor.detach().view(bit_dtype).cpu().numpy().tobytes(order="C")


def digest(tensor):
    return hashlib.sha256(raw(tensor)).hexdigest()


def preserve_failure(output, name, tensors):
    directory = output.parent / (output.stem + "-failure")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"case": name, "storage_note": "complete logical ND contiguous bytes", "tensors": {}}
    for key, tensor in tensors.items():
        data = raw(tensor)
        (directory / (key + ".bin")).write_bytes(data)
        metadata["tensors"][key] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return str(directory)


def check_quantize(module, parameter_cases, device, output, report):
    quantize = module.quantize
    resources = module.get_device_resources(device.index)
    report["tiling_resources"] = resources._asdict()
    held_graphs = []

    def oracle(x, reciprocal, offset):
        width = x.shape[-1] if x.ndim else 1
        return torch_npu.npu_quantize(
            x,
            reciprocal.expand(width).contiguous() if reciprocal.numel() == 1 else reciprocal,
            offset.expand(width).contiguous() if offset.numel() == 1 else offset,
            torch.qint8,
            -1,
            False,
        )

    def check(name, x, reciprocal, offset, implementation=None, extra=None):
        report["phase"] = name
        tensors = {"input": x, "reciprocal": reciprocal, "offset": offset, **(extra or {})}
        before = {key: digest(value) for key, value in tensors.items()}
        try:
            report["operation"] = "original_cann_oracle"
            expected = oracle(x, reciprocal, offset)
            tensors["expected"] = expected
            report["operation"] = "triton_candidate" if implementation is None else "captured_output"
            actual = quantize(x, reciprocal, offset) if implementation is None else implementation()
            tensors["actual"] = actual
            report["operation"] = "strict_compare"
            torch.npu.synchronize()
            expected_bytes, actual_bytes = raw(expected), raw(actual)
            unchanged = all(digest(tensors[key]) == value for key, value in before.items())
            passed = (
                actual.dtype == torch.int8
                and actual.device == x.device
                and torch_npu.get_npu_format(actual) == 2
                and actual.shape == expected.shape
                and actual.is_contiguous()
                and actual_bytes == expected_bytes
                and unchanged
            )
            report["checks"].append(
                {
                    "case": name,
                    "dtype": str(x.dtype),
                    "shape": list(x.shape),
                    "stride": list(x.stride()),
                    "parameter_shape": list(reciprocal.shape),
                    "parameter_dtype": str(reciprocal.dtype),
                    "parameter_strides": [list(reciprocal.stride()), list(offset.stride())],
                    "input_unchanged": unchanged,
                    "mismatched_bytes": sum(a != b for a, b in zip(actual_bytes, expected_bytes)),
                    "actual_bytes": len(actual_bytes),
                    "expected_bytes": len(expected_bytes),
                    "input_sha256": before["input"],
                    "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                    "pass": passed,
                }
            )
            if not passed:
                raise AssertionError(report["checks"][-1])
            return report["checks"][-1]["expected_sha256"]
        except Exception:
            report["failure_artifact"] = preserve_failure(output, name, tensors)
            raise

    cases = [
        {"name": "unit", "scale_recip": 1.0, "offset": 0.0},
        {"name": "half_ties", "scale_recip": 1.0, "offset": 0.5},
        {"name": "non_power", "scale_recip": 3.140625, "offset": -3.25},
        {"name": "small", "scale_recip": 0.0078125, "offset": 3.0},
    ] + parameter_cases
    shapes = (
        (0, 7),
        (1, 1),
        (1, 2048),
        (1, 8192),
        (2, 3),
        (6, 2048),
        (6, 8192),
        (17, 129),
        (33, 8193),
        # Exercise core-owned multi-row batches, their tails, and UB-split
        # wide rows; small row counts alone never enter the batch loop.
        (67, 17),
        (513, 17),
        (128, 8192),
        (128,),
        (2, 3, 127),
        (2, 1, 4, 33),
        (1, 2, 1, 1, 1, 1, 3, 7),
    )
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    storage = report["selected_storage"]
    for label in report["selected_dtypes"]:
        dtype = dtypes[label]
        report["dtype_status"][label] = "running"
        if storage != "nz":
            element_bytes = torch.empty((), dtype=dtype).element_size()
            # Cross both one aligned wave and a raw UB-sized input. These are
            # device-derived coverage boundaries, not copies of the planner.
            boundary_shapes = tuple(
                (elements + delta,)
                for elements in (
                    resources.vector_cores * module.DMA_ALIGNMENT_BYTES // element_bytes,
                    resources.ub_bytes // element_bytes,
                )
                for delta in (-1, 0, 1)
            ) + (
                (3, resources.ub_bytes // element_bytes + 1),
                (2, resources.vector_cores * resources.ub_bytes // element_bytes + 1),
            )
            if dtype == torch.float32:
                bits = (torch.arange(65536, dtype=torch.int64) * 65537).to(torch.int32)
                pattern_name = "65536_stratified_fp32_patterns_not_exhaustive"
            else:
                bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
                pattern_name = "all65536_16bit_patterns"
            patterns = bits.view(dtype).reshape(8, 8192)
            all_values = patterns.to(device)
            if raw(all_values) != raw(patterns):
                report["failure_artifact"] = preserve_failure(
                    output, label + "/pattern_h2d", {"cpu_patterns": patterns, "device_patterns": all_values}
                )
                raise AssertionError("Pattern H2D did not preserve every input bit")
            for case in cases:
                reciprocal = torch.tensor([case["scale_recip"]], device=device, dtype=dtype)
                offset = torch.tensor([case["offset"]], device=device, dtype=dtype)
                check(label + "/" + pattern_name + "/" + str(case["name"]), all_values, reciprocal, offset)
                channel = reciprocal.expand(all_values.shape[-1]).contiguous()
                channel_offset = offset.expand(all_values.shape[-1]).contiguous()
                check(label + "/" + pattern_name + "/channel/" + str(case["name"]), all_values, channel, channel_offset)
                # Retain the previously exposed FP32 scalar-parameter API.
                prepared = reciprocal.float(), offset.float()
                check(
                    label + "/" + pattern_name + "/fp32_scalar/" + str(case["name"]),
                    all_values,
                    reciprocal,
                    offset,
                    implementation=lambda all_values=all_values, prepared=prepared: quantize(all_values, *prepared),
                    extra={"prepared_reciprocal": prepared[0], "prepared_offset": prepared[1]},
                )

            reciprocal = torch.tensor([1.0], device=device, dtype=dtype)
            offset = torch.tensor([0.0], device=device, dtype=dtype)
            for shape in dict.fromkeys((*shapes, *boundary_shapes)):
                for layout in ("contiguous", "sliced", "transpose", "expanded"):
                    if layout == "sliced":
                        backing = (torch.randn((*shape[:-1], shape[-1] * 2 + 3)) * 32).to(device=device, dtype=dtype)
                        x = backing[..., 1 : 1 + 2 * shape[-1] : 2]
                    elif layout == "expanded":
                        backing = (torch.randn((1,) * len(shape)) * 32).to(device=device, dtype=dtype)
                        x = backing.expand(shape)
                    else:
                        backing = (torch.randn(shape) * 32).to(device=device, dtype=dtype)
                        x = backing
                        if layout == "transpose" and len(shape) > 1:
                            x = backing.movedim(-1, 0).contiguous().movedim(0, -1)
                    check(label + "/" + str(shape) + "/" + layout, x, reciprocal, offset, extra={"backing": backing})
                    # Different values by column expose wrong broadcast/column
                    # indexing; sliced parameter vectors exercise real strides.
                    column = torch.arange(shape[-1] * 2, dtype=torch.float32)
                    channel_backing = (0.25 + (column % 7) / 8).to(device=device, dtype=dtype)
                    offset_backing = ((column % 9) / 4 - 1).to(device=device, dtype=dtype)
                    channel, channel_offset = channel_backing[::2], offset_backing[1::2]
                    check(
                        label + "/" + str(shape) + "/" + layout + "/strided_channel",
                        x,
                        channel,
                        channel_offset,
                        extra={
                            "backing": backing,
                            "channel_backing": channel_backing,
                            "offset_backing": offset_backing,
                        },
                    )
                    if shape in ((67, 17), (513, 17), (128, 8192)):
                        check(
                            label + "/" + str(shape) + "/" + layout + "/contiguous_channel",
                            x,
                            channel.contiguous(),
                            channel_offset.contiguous(),
                            extra={"backing": backing},
                        )

            # Physical-span loads must neither overread a row tail nor confuse
            # sparse parameter lanes. Exercise both sides of the DMA-span
            # decision, non-power-of-two strides, and nonzero storage offsets.
            for step in (3, 5, 16, 17):
                for shape in ((1, 1), (7, 33), (17, 257)):
                    backing = (torch.randn((*shape[:-1], shape[-1] * step + 1)) * 32).to(device=device, dtype=dtype)
                    x = backing[..., 1 : 1 + shape[-1] * step : step]
                    channels = torch.arange(shape[-1] * step + 1, dtype=torch.float32)
                    scale_storage = (0.25 + channels % 7 / 8).to(device=device, dtype=dtype)
                    offset_storage = (channels % 9 / 4 - 1).to(device=device, dtype=dtype)
                    check(
                        label + "/span/" + str(shape) + "/stride" + str(step),
                        x,
                        scale_storage[1::step],
                        offset_storage[1::step],
                        extra={"backing": backing, "scale_storage": scale_storage, "offset_storage": offset_storage},
                    )

            # A physically contiguous middle axis can coexist with a
            # broadcast leading dimension. This enters the ND transpose copy
            # for per-tensor parameters and must still read current storage.
            backing = (torch.randn(1, 17, 5) * 32).to(device=device, dtype=dtype)
            mixed = backing.transpose(1, 2).expand(3, 5, 17)
            check(label + "/mixed_broadcast_transpose/scalar", mixed, reciprocal, offset, extra={"backing": backing})
            check(
                label + "/mixed_broadcast_transpose/channel",
                mixed,
                reciprocal.expand(17),
                offset.expand(17),
                extra={"backing": backing},
            )

            ties = torch.arange(-130, 131, dtype=torch.float32) + 0.5
            nearby = torch.stack(
                (
                    torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
                    ties,
                    torch.nextafter(ties, torch.full_like(ties, float("inf"))),
                )
            )
            check(label + "/half_neighbors", nearby.to(device=device, dtype=dtype), reciprocal, offset)

        if storage != "nd":
            reciprocal = torch.tensor([1.0], device=device, dtype=dtype)
            offset = torch.tensor([0.0], device=device, dtype=dtype)
            physical = (torch.randn(6, 129) * 32).to(device=device, dtype=dtype)
            report["phase"] = label + "/outside_contract_nz_probe"
            previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
            previous = previous.decode() if isinstance(previous, bytes) else previous
            if previous not in ("enable", "disable"):
                raise RuntimeError(f"Cannot safely restore ALLOW_INTERNAL_FORMAT: {previous!r}")
            probe = {
                "dtype": label,
                "before": previous,
                "requested_format": 29,
                "valid_original_qint8_domain": False,
                "status": "running",
            }
            report.setdefault("nz_diagnostics", []).append(probe)
            try:
                torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": "enable"})
                blocked = torch_npu.npu_format_cast(physical, 29)
                probe["actual_format"] = torch_npu.get_npu_format(blocked)
                if probe["actual_format"] != 29:
                    probe["status"] = "unsupported"
                    raise RuntimeError(f"Requested NZ input was not created: {probe}")
                try:
                    native = oracle(blocked, reciprocal, offset)
                    data = raw(native)
                    probe["oracle_result"] = {
                        "dtype": str(native.dtype),
                        "shape": list(native.shape),
                        "format": torch_npu.get_npu_format(native),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "first48_bytes_hex": data[:48].hex(),
                    }
                except Exception as error:
                    probe["oracle_error"] = repr(error)
                try:
                    quantize(blocked, reciprocal.float(), offset.float())
                except ValueError as error:
                    probe["candidate_rejected"] = True
                    probe["candidate_error"] = str(error)
                else:
                    raise AssertionError("QINT8 candidate must reject unsupported physical NZ storage")
                probe["status"] = "diagnostic_complete_not_quantization_pass"
            finally:
                torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous})
                restored = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
                probe["after"] = restored.decode() if isinstance(restored, bytes) else restored
                if probe["after"] != previous:
                    raise RuntimeError(f"ALLOW_INTERNAL_FORMAT restoration failed: {probe}")

        if storage != "nz":
            for parameter_layout in ("scalar_fp32", "scalar_native", "channel", "strided_channel"):
                # Every replay uses a fresh CANN oracle. Mutate input and parameter
                # tensors in place to catch cached scalar values or stale copies.
                backing = (torch.randn(6, 134) * 32).to(device=device, dtype=dtype)
                original, changed = backing.clone(), -backing + 3
                x = backing[:, :129]
                parameter_width = 1 if parameter_layout.startswith("scalar_") else 129
                step = 2 if parameter_layout == "strided_channel" else 1
                reciprocal = (
                    0.5 + torch.arange(parameter_width * step, device=device, dtype=torch.float32) % 5 / 8
                ).to(dtype)[::step]
                offset = (torch.arange(parameter_width * step, device=device, dtype=torch.float32) % 7 / 4 - 1).to(
                    dtype
                )[::step]
                original_reciprocal, original_offset = reciprocal.clone(), offset.clone()
                changed_reciprocal, changed_offset = reciprocal * 2, offset + 3
                prepared = (
                    (reciprocal.float(), offset.float()) if parameter_layout == "scalar_fp32" else (reciprocal, offset)
                )
                outputs, graphs = {}, {}
                for mode in ("cann", "triton"):
                    invoke = (
                        (lambda x=x, reciprocal=reciprocal, offset=offset: oracle(x, reciprocal, offset))
                        if mode == "cann"
                        else (lambda x=x, prepared=prepared: quantize(x, *prepared))
                    )
                    invoke()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs[mode] = invoke()
                    graphs[mode] = graph
                held_graphs.append(
                    (
                        graphs,
                        outputs,
                        backing,
                        original,
                        changed,
                        prepared,
                        original_reciprocal,
                        original_offset,
                        changed_reciprocal,
                        changed_offset,
                    )
                )
                torch.npu.synchronize()
                initial_sha = None
                addresses = (reciprocal.data_ptr(), offset.data_ptr())
                for stage, values, scale_values, offset_values in (
                    ("unchanged", original, original_reciprocal, original_offset),
                    ("changed_input", changed, original_reciprocal, original_offset),
                    ("changed_reciprocal", original, changed_reciprocal, original_offset),
                    ("changed_offset", original, original_reciprocal, changed_offset),
                    ("restored", original, original_reciprocal, original_offset),
                ):
                    backing.copy_(values)
                    reciprocal.copy_(scale_values)
                    offset.copy_(offset_values)
                    if parameter_layout == "scalar_fp32":
                        prepared[0].copy_(reciprocal)
                        prepared[1].copy_(offset)
                    if (reciprocal.data_ptr(), offset.data_ptr()) != addresses:
                        raise AssertionError("Graph parameter addresses must remain unchanged")
                    stage_sha = None
                    for mode in ("cann", "triton"):
                        for repeat in range(2):

                            def replay(mode=mode, graphs=graphs, outputs=outputs):
                                graphs[mode].replay()
                                return outputs[mode]

                            stage_sha = check(
                                label + "/graph/" + parameter_layout + "/" + stage + "/" + mode + "/" + str(repeat),
                                x,
                                reciprocal,
                                offset,
                                implementation=replay,
                                extra={
                                    "backing": backing,
                                    "original": original,
                                    "changed": changed,
                                    "prepared_reciprocal": prepared[0],
                                    "prepared_offset": prepared[1],
                                },
                            )
                    if stage == "unchanged":
                        initial_sha = stage_sha
                    elif (stage.startswith("changed_") and stage_sha == initial_sha) or (
                        stage == "restored" and stage_sha != initial_sha
                    ):
                        report["failure_artifact"] = preserve_failure(
                            output,
                            label + "/graph/input_freshness",
                            {"input": x, "backing": backing, "original": original, "changed": changed},
                        )
                        raise AssertionError("Graph input/parameter freshness/restoration witness failed")
        report["dtype_status"][label] = "passed" if storage != "nz" else "diagnostic_only_not_nd_tested"
        print(
            json.dumps(
                {
                    "completed_dtype": label,
                    "status": report["dtype_status"][label],
                    "checks_completed": len(report["checks"]),
                }
            ),
            flush=True,
        )
    report["retained_graphs"] = len(held_graphs) * 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-path", required=True, type=Path)
    parser.add_argument("--parameter-json", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--storage", choices=("nd", "nz", "all"), default="nd")
    parser.add_argument("--dtype", action="append", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "checks": [],
        "selected_storage": args.storage,
        "selected_dtypes": args.dtype or ["bf16", "fp16", "fp32"],
        "dtype_status": {
            k: "pending" if args.dtype is None or k in args.dtype else "not_selected" for k in ("bf16", "fp16", "fp32")
        },
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "kernel_sha256": hashlib.sha256(args.kernel_path.read_bytes()).hexdigest(),
        "parameter_sha256": hashlib.sha256(args.parameter_json.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Original QINT8 ND equality only; direct NZ is outside-contract diagnostics, not a pass",
        "contract_source": (
            "https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/"
            "context/ops-nn/aclnnAscendQuantV3.md"
        ),
    }
    try:
        torch.npu.set_device(args.device)
        # Match the model process, including its Triton utility replacements.
        importlib.import_module("vllm_ascend.patch.worker")
        import triton

        report["triton_next_power_of_2"] = triton.next_power_of_2.__module__
        torch.manual_seed(20261028)
        parameters = json.loads(args.parameter_json.read_text())
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("Actual per-tensor model parameter cases are required")
        report["parameter_cases"] = len(parameters)
        spec = importlib.util.spec_from_file_location("static_quantize_under_test", args.kernel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        resource_module = sys.modules[module.get_device_resources.__module__]
        resource_path = Path(resource_module.__file__).resolve()
        expected_resource = args.kernel_path.parent / "kernel_resources.py"
        resource_sha = hashlib.sha256(resource_path.read_bytes()).hexdigest()
        expected_resource_sha = hashlib.sha256(expected_resource.read_bytes()).hexdigest()
        report["resource_module"] = {
            "imported_path": str(resource_path),
            "imported_sha256": resource_sha,
            "candidate_path": str(expected_resource.resolve()),
            "candidate_sha256": expected_resource_sha,
        }
        if resource_sha != expected_resource_sha:
            raise RuntimeError("Imported kernel_resources does not match the candidate sibling file")
        with torch.inference_mode():
            check_quantize(module, parameters, torch.device("npu", args.device), args.output, report)
        report["nd_numerical_checks_passed"] = (
            args.storage != "nz" and bool(report["checks"]) and all(c["pass"] for c in report["checks"])
        )
        report["status"] = (
            "passed"
            if args.storage == "nd"
            else "passed_nd_with_nz_diagnostics"
            if args.storage == "all"
            else "nz_diagnostic_completed_no_valid_domain_pass"
        )
    except Exception as error:
        report["status"] = "failed"
        for label in report["dtype_status"]:
            if report["dtype_status"][label] == "running":
                report["dtype_status"][label] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"status": report["status"], "checks": len(report["checks"]), "dtype_status": report["dtype_status"]}
        )
    )
