# SPDX-License-Identifier: Apache-2.0
"""Explicit NPU bit gate for V-up; optional timing starts after the whole gate.

No model/service control. Importing this file or --self-test never imports torch.
Default cases are positive-dimension ND inputs; empty/NZ need explicit --case.
New dtype, stride and reduction domains remain unvalidated until this runs.
--cube-math-type overrides only this test process, for both implementations.
--hf32-sequence-check adds frozen-oracle mode/stream checks after the original gate.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import struct
import sys
import time
import traceback
from pathlib import Path

CASES = (
    (
        ("legacy", (6, 32, 512, 256), "contiguous"),
        ("scalar", (1, 1, 1, 1), "contiguous"),
        ("tails", (7, 3, 33, 65), "contiguous"),
        ("multi_k", (17, 2, 1025, 259), "contiguous"),
        ("m_tile_boundary", (33, 3, 513, 129), "contiguous"),
        ("large_m_tile", (129, 2, 65, 33), "contiguous"),
        ("strided", (3, 2, 35, 19), "strided"),
        ("transposed", (4, 3, 64, 33), "transposed"),
        ("strided_dense_suffix", (6, 32, 512, 256), "strided_dense_suffix"),
        ("transpose_reverse", (6, 32, 512, 256), "transpose_reverse"),
        ("n0", (0, 2, 17, 7), "contiguous"),
        ("h0", (2, 0, 17, 7), "contiguous"),
        ("v0", (2, 2, 17, 0), "contiguous"),
        ("k0", (2, 3, 0, 7), "contiguous"),
        ("legacy_nz", (6, 32, 512, 256), "nz"),
        ("special", (3, 2, 33, 17), "special"),
        ("nonfinite", (2, 2, 33, 17), "nonfinite"),
        ("mtp_n1", (1, 32, 512, 256), "contiguous"),
    )
    + tuple(
        (f"{operand}_zero_{axis}", (3, 2, 33, 17), f"{operand}_zero_{axis}")
        for operand in ("x", "weight")
        for axis in range(3)
    )
    + (
        ("cancel_kv", (6, 3, 1025, 259), "cancel_early"),
        ("cancel_mkv", (257, 2, 1025, 259), "cancel_late"),
    )
)

HF32_ROUNDING_BITS = (
    0x3F8007FF,
    0x3F800800,
    0x3F800801,
    0x3F801800,
    0xBF8007FF,
    0xBF800800,
    0xBF800801,
    0xBF801800,
    0x3F800000,
)


def copy_bit_payload(dtype_name, physical_count, step):
    """Repeat each logical raw pattern across its physical stride slots."""
    code = "I" if dtype_name == "fp32" else "H"
    patterns = (
        (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FA12345, 0xFFFFFFFF, 0x3F800001)
        if dtype_name == "fp32"
        else range(65536)
    )
    return b"".join(struct.pack("<" + code, patterns[(i // step) % len(patterns)]) for i in range(physical_count))


def cancellation_terms(k, kind):
    """Finite, exactly representable half inputs; products cancel across K tiles.

    32768 * 512 is 2**24: adding 1 before cancellation can lose the residual
    in FP32. CANN is still the sole output oracle, not this arithmetic example.
    """
    products = [(32768.0, 512.0), (1.0, 1.0), (-32768.0, 512.0), (0.5, 1.0), (0.25, 1.0)]
    if kind == "cancel_late":
        products[1], products[2] = products[2], products[1]
    positions = (0, k // 4, k // 2, 3 * k // 4, k - 1)
    return tuple((position, x, weight) for position, (x, weight) in zip(positions, products))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_control_artifacts(source, expected_bitcode, expected_source, record):
    """Bind the two fixed companion paths when this runtime supplies either one."""
    artifacts = (
        ("bitcode", source.with_name("vup_hf32_control.bc"), expected_bitcode),
        ("source", source.with_name("vup_hf32_control.cpp"), expected_source),
    )
    required = any(path.exists() or expected is not None for _, path, expected in artifacts)
    record.update(required=required, status="checking", files={})
    for name, path, expected in artifacts:
        record["files"][name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": sha(path) if path.is_file() else None,
        }
    if not required:
        record["status"] = "not_present"
        return
    for name, _, expected in artifacts:
        actual = record["files"][name]["actual_sha256"]
        if not expected or actual is None or actual != expected.lower():
            record["status"] = "failed"
            raise ValueError(f"HF32 control {name} is missing, changed or lacks its expected SHA-256")
    record["status"] = "passed"


def check_resources_artifact(source, expected_sha, module, record):
    """Require the adjacent resource source and bind the helper actually imported."""
    adjacent = source.with_name("kernel_resources.py").resolve()
    record.update(expected_path=str(adjacent), expected_sha256=expected_sha, status="checking")
    record["adjacent_sha256"] = sha(adjacent) if adjacent.is_file() else None
    if not expected_sha or record["adjacent_sha256"] != expected_sha.lower():
        record["status"] = "failed"
        raise ValueError("Resource helper is missing, changed or lacks its expected SHA-256")
    if module is not None:
        imported = sys.modules[module.get_device_resources.__module__]
        actual = Path(imported.__file__).resolve(strict=True)
        record.update(actual_import_path=str(actual), actual_import_sha256=sha(actual))
        if actual != adjacent or record["actual_import_sha256"] != expected_sha.lower():
            record["status"] = "failed"
            raise ValueError("Candidate imported a different resource helper path or version")
    record["status"] = "passed"


def cube_math_snapshot(torch_npu):
    """Observe options without changing them; shared by the mode leakage gate."""
    matmul = torch_npu.npu.matmul
    options = {}
    for key in ("CUBE_MATH_TYPE", "ALLOW_MATMUL_HF32", "ACL_PRECISION_MODE"):
        value = torch_npu._C._npu_getOption(key)
        options[key] = value.decode() if isinstance(value, bytes) else value
    selected = matmul.cube_math_type
    return {
        "cube_math_type": None if selected is None else int(selected),
        "allow_hf32": matmul.allow_hf32,
        "options": options,
    }


def configure_cube_math_type(torch_npu, requested, report):
    """Keep inherited settings unless the caller explicitly selects an enum."""
    matmul = torch_npu.npu.matmul
    settings = report["math_mode"] = {"override_requested": requested, "initial": cube_math_snapshot(torch_npu)}
    if requested is not None:
        matmul.cube_math_type = torch_npu.npu.CubeMathType(requested)
    settings["effective"] = cube_math_snapshot(torch_npu)
    if requested is not None and settings["effective"]["cube_math_type"] != requested:
        raise RuntimeError("Explicit cube_math_type did not take effect")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int)
    parser.add_argument("--kernel-path")
    parser.add_argument("--expected-kernel-sha256")
    parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--expected-resources-sha256")
    parser.add_argument("--expected-control-bitcode-sha256")
    parser.add_argument("--expected-control-source-sha256")
    parser.add_argument("--output")
    parser.add_argument("--dtype", action="append", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--case",
        action="append",
        choices=[case[0] for case in CASES],
        help="Explicit case, including empty/NZ diagnostics; defaults to all positive-dimension ND cases",
    )
    parser.add_argument("--oracle", choices=("transpose", "bmm"), default="transpose")
    parser.add_argument(
        "--cube-math-type",
        type=int,
        choices=range(5),
        help="Explicit process-local CubeMathType override (0..4); omitted keeps inherited settings",
    )
    parser.add_argument(
        "--hf32-sequence-check",
        action="store_true",
        help="Add strict mode0/mode3 eager/graph/stream checks; requires --cube-math-type for exact restoration",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--graph-repeats", type=int, default=2)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--raw-copy-only",
        action="store_true",
        help="Check only original raw-bit copies; does not validate BMM arithmetic or performance",
    )
    parser.add_argument("--timing-case", action="append", choices=[case[0] for case in CASES])
    parser.add_argument("--samples", type=int, default=20, help="ABBA blocks per timing case/scope")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        from enum import IntEnum
        from types import SimpleNamespace
        from unittest.mock import patch

        for dtype_name in ("bf16", "fp16", "fp32"):
            code = "I" if dtype_name == "fp32" else "H"
            expected_patterns = (
                {0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FA12345, 0xFFFFFFFF, 0x3F800001}
                if dtype_name == "fp32"
                else set(range(65536))
            )
            for step in (1, 2):
                payload = copy_bit_payload(dtype_name, 65536 * step, step)
                words = [item[0] for item in struct.iter_unpack("<" + code, payload)]
                assert set(words[::step]) == expected_patterns
        assert len(CASES) == len({name for name, _, _ in CASES})
        assert all(len(shape) == 4 and min(shape) >= 0 for _, shape, _ in CASES)
        assert {kind for _, _, kind in CASES} >= {"strided", "transposed", "nz", "x_zero_2", "weight_zero_2"}
        valid_cases = [name for name, shape, kind in CASES if all(shape) and kind != "nz"]
        assert {"mtp_n1", "m_tile_boundary", "large_m_tile"}.issubset(valid_cases)
        assert {"cancel_kv", "cancel_mkv"}.issubset(valid_cases)
        assert not {"n0", "h0", "v0", "k0", "legacy_nz"}.intersection(valid_cases)
        rounded_sums = []
        for name, shape, kind in CASES:
            if not kind.startswith("cancel_"):
                continue
            terms = cancellation_terms(shape[2], kind)
            assert len({position for position, _, _ in terms}) == len(terms)
            assert min(position for position, _, _ in terms) == 0
            assert max(position for position, _, _ in terms) == shape[2] - 1
            total = 0.0
            for _, x, weight in terms:
                for value in (x, weight):
                    assert math.isfinite(value)
                    assert struct.unpack("<e", struct.pack("<e", value))[0] == value
                    assert struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF == 0
                total = struct.unpack("<f", struct.pack("<f", total + x * weight))[0]
            assert math.fsum(x * weight for _, x, weight in terms) == 1.75
            rounded_sums.append(total)
        # The fixture is sensitive to FP32 accumulation order; this is not a
        # prescribed NPU answer and does not relax the exact native comparison.
        assert rounded_sums == [0.75, 1.75]
        enum = IntEnum("_CubeMathType", {f"MODE_{mode}": mode for mode in range(5)})
        assert parser.parse_args([]).cube_math_type is None
        assert not parser.parse_args([]).hf32_sequence_check
        assert parser.parse_args(["--hf32-sequence-check", "--cube-math-type", "3"]).hf32_sequence_check
        assert len(HF32_ROUNDING_BITS) == 9
        assert {bits & 0xFFF for bits in HF32_ROUNDING_BITS} >= {0x7FF, 0x800, 0x801}
        assert all((bits ^ 0x80000000) in HF32_ROUNDING_BITS for bits in HF32_ROUNDING_BITS[:4])
        parsed_controls = parser.parse_args(
            ["--expected-control-bitcode-sha256", "a" * 64, "--expected-control-source-sha256", "b" * 64]
        )
        assert parsed_controls.expected_control_bitcode_sha256 == "a" * 64
        assert parsed_controls.expected_control_source_sha256 == "b" * 64
        control_files = {}

        class ControlPath:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

            def exists(self):
                return self.name in control_files

            is_file = exists
            with_name = lambda self, name: ControlPath(name)

        with patch(__name__ + ".sha", side_effect=lambda path: hashlib.sha256(control_files[str(path)]).hexdigest()):
            candidate = ControlPath("vup_bmm.py")
            bitcode, control_source = "vup_hf32_control.bc", "vup_hf32_control.cpp"
            record = {}
            check_control_artifacts(candidate, None, None, record)
            assert record["status"] == "not_present"
            control_files[bitcode] = b"control-bitcode"
            control_files[control_source] = b"control-source"
            bc_sha, cpp_sha = sha(bitcode), sha(control_source)
            check_control_artifacts(candidate, bc_sha, cpp_sha, record)
            assert record["status"] == "passed"
            for expected_bc, expected_cpp in (
                (None, None),
                (bc_sha, None),
                (None, cpp_sha),
                ("0" * 64, cpp_sha),
                (bc_sha, "0" * 64),
            ):
                try:
                    check_control_artifacts(candidate, expected_bc, expected_cpp, record)
                except ValueError:
                    assert record["status"] == "failed"
                else:
                    raise AssertionError("Missing/changed companion identity was accepted")
            control_files[bitcode] = b"changed-bitcode"
            try:
                check_control_artifacts(candidate, bc_sha, cpp_sha, record)
            except ValueError:
                assert record["files"]["bitcode"]["actual_sha256"] != bc_sha
            else:
                raise AssertionError("Changed bitcode was accepted")
        adjacent = Path("candidate/kernel_resources.py").resolve()
        different = Path("installed/kernel_resources.py").resolve()
        resource_sha = "c" * 64
        get_resources = lambda device: None
        get_resources.__module__ = "_vup_resources_test"
        candidate = SimpleNamespace(get_device_resources=get_resources)
        imported = SimpleNamespace(__file__=str(adjacent))
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "resolve", lambda self, **kwargs: self.absolute()),
            patch(__name__ + ".sha", return_value=resource_sha),
            patch.dict(sys.modules, {"_vup_resources_test": imported}),
        ):
            source = adjacent.with_name("vup_bmm.py")
            record = {}
            check_resources_artifact(source, resource_sha, candidate, record)
            assert record["status"] == "passed" and record["actual_import_path"] == str(adjacent)
            for expected, actual_path in ((None, adjacent), ("0" * 64, adjacent), (resource_sha, different)):
                imported.__file__ = str(actual_path)
                try:
                    check_resources_artifact(source, expected, candidate, record)
                except ValueError:
                    assert record["status"] == "failed"
                else:
                    raise AssertionError("Missing/changed/mixed resource helper was accepted")
        for requested in (None, *range(5)):
            matmul = SimpleNamespace(cube_math_type=enum(3), allow_hf32=False)

            def get_option(key, matmul=matmul):
                return {
                    "CUBE_MATH_TYPE": str(int(matmul.cube_math_type)).encode(),
                    "ALLOW_MATMUL_HF32": b"disable",
                    "ACL_PRECISION_MODE": b"must_keep_origin_dtype",
                }[key]

            mock = SimpleNamespace(
                npu=SimpleNamespace(matmul=matmul, CubeMathType=enum),
                _C=SimpleNamespace(_npu_getOption=get_option),
            )
            settings = {}
            configure_cube_math_type(mock, requested, settings)
            assert settings["math_mode"]["initial"]["cube_math_type"] == 3
            assert isinstance(matmul.cube_math_type, enum)
            assert settings["math_mode"]["effective"]["cube_math_type"] == (3 if requested is None else requested)
            assert settings["math_mode"]["effective"]["allow_hf32"] is False
            if requested is not None:
                assert parser.parse_args(["--cube-math-type", str(requested)]).cube_math_type == requested
        print(
            json.dumps(
                {
                    "cpu_self_test": "passed",
                    "cases": len(CASES),
                    "default_valid_cases": len(valid_cases),
                    "cube_math_type_cases": 6,
                    "npu_run": False,
                    "hf32_sequence_npu_status": "pending",
                }
            )
        )
        return
    if args.raw_copy_only and (args.timing or args.hf32_sequence_check):
        parser.error("--raw-copy-only cannot be combined with BMM timing or HF32 sequence checks")
    if args.hf32_sequence_check and (args.cube_math_type is None or args.oracle != "transpose"):
        parser.error("--hf32-sequence-check requires explicit --cube-math-type and the original transpose oracle")
    if (
        args.device is None
        or args.device < 0
        or not args.output
        or not args.kernel_path
        or not args.expected_kernel_sha256
        or not args.expected_resources_sha256
        or args.graph_repeats < 2
        or args.samples < 20
        or sys.byteorder != "little"
    ):
        parser.error(
            "Explicit NPU, new output, kernel/resource SHA, >=2 graph repeats, >=20 samples and little endian required"
        )
    source = Path(args.kernel_path).resolve(strict=True)
    if sha(source) != args.expected_kernel_sha256.lower():
        parser.error("Kernel SHA mismatch before torch import")
    if args.expected_driver_sha256 and sha(__file__) != args.expected_driver_sha256.lower():
        parser.error("Driver SHA mismatch before torch import")
    if args.raw_copy_only:
        selected = []
    elif args.case is None:
        selected = [case for case in CASES if all(case[1]) and case[2] != "nz"]
    else:
        selected = [case for case in CASES if case[0] in args.case]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "arguments": vars(args),
        "driver_sha256": sha(__file__),
        "kernel_sha256": sha(source),
        "checks": [],
        "copy_bit_checks": [],
        "timings": [],
        "cases": [],
        "selected_cases": [case[0] for case in selected],
        "selected_dtypes": args.dtype or ["bf16", "fp16", "fp32"],
        "numerical_checks_passed": False,
        "performance_results_valid": False,
        "scope": (
            "Raw-bit normalization only; no BMM arithmetic or performance validation"
            if args.raw_copy_only
            else "Logical [N,H,V] bits and immutable full ND input buffers; empty/NZ are explicit domain diagnostics"
        ),
        "explicit_domain_diagnostics": any(not all(shape) or kind == "nz" for _, shape, kind in selected),
        "layout_note": "Candidate output is contiguous ND; reference strides/physical format are recorded separately",
        "oracle_note": "No oracle fallback: an unsupported selected CANN domain fails this gate",
        "npu_domains_pending": True,
        "hf32_sequence": {
            "requested": args.hf32_sequence_check,
            "status": "pending" if args.hf32_sequence_check else "not_requested",
        },
        "control_artifact_checks": [],
        "resource_artifact_checks": [],
    }
    last, keep_alive, timing_jobs = {}, [], []

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    def verify_control(phase):
        record = {"phase": phase}
        report["control_artifact_checks"].append(record)
        check_control_artifacts(
            source, args.expected_control_bitcode_sha256, args.expected_control_source_sha256, record
        )

    def verify_resources(phase, module=None):
        record = {"phase": phase}
        report["resource_artifact_checks"].append(record)
        check_resources_artifact(source, args.expected_resources_sha256, module, record)

    save()
    try:
        verify_control("before_torch_import")
        verify_resources("before_torch_import")
        save()
        import torch
        import torch_npu

        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        # Match the model process, including its Triton utility replacements.
        importlib.import_module("vllm_ascend.patch.worker")
        import triton

        report["triton_next_power_of_2"] = triton.next_power_of_2.__module__
        device = torch.device("npu", args.device)
        configure_cube_math_type(torch_npu, args.cube_math_type, report)
        spec = importlib.util.spec_from_file_location("vup_bmm_candidate", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if sha(source) != args.expected_kernel_sha256.lower() or not callable(getattr(module, "vup_bmm", None)):
            raise ValueError("Candidate changed while importing or lacks vup_bmm")
        verify_resources("after_candidate_import", module)
        verify_control("after_candidate_import")
        report["device_resources"] = module.get_device_resources(args.device)._asdict()
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        matmul = getattr(torch.npu, "matmul", None)
        try:
            report["allow_hf32_matmul"] = getattr(matmul, "allow_hf32", None)
        except Exception as error:
            report["allow_hf32_matmul"] = {"unavailable": repr(error)}
        getter = getattr(getattr(torch_npu, "_C", None), "_npu_getOption", None)
        if callable(getter):
            try:
                option = getter("ALLOW_MATMUL_HF32")
                report["ALLOW_MATMUL_HF32_option"] = option.decode() if isinstance(option, bytes) else option
            except Exception as error:
                report["ALLOW_MATMUL_HF32_option"] = {"unavailable": repr(error)}
        report["precision_note"] = (
            "Inherited settings are unchanged unless --cube-math-type explicitly overrides this test process. "
            "The same effective settings apply to CANN and candidate; all output bits still require equality. "
            "A null cube_math_type property means no explicit override, not effective CANN mode zero."
        )
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

        def raw(tensor):
            snapshot = tensor.detach()
            if snapshot.device.type == "npu" and torch_npu.get_npu_format(snapshot) != 2:
                snapshot = torch_npu.npu_format_cast(snapshot, 2)
                if torch_npu.get_npu_format(snapshot) != 2:
                    raise RuntimeError("Raw snapshot conversion did not produce ND")
            integer_dtype = {2: torch.int16, 4: torch.int32}[snapshot.element_size()]
            return snapshot.view(integer_dtype).cpu().numpy().tobytes(order="C")

        def metadata(tensor):
            return {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "storage_offset": tensor.storage_offset(),
                "npu_format": torch_npu.get_npu_format(tensor),
            }

        def random_cpu(shape, dtype, seed):
            return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.25).to(dtype)

        def operand(shape, kind, role, dtype_name, seed):
            dtype = dtypes[dtype_name]
            if kind.startswith("cancel_"):
                data = torch.zeros(shape, dtype=dtype)
                k = shape[2] if role == "x" else shape[1]
                # Alternate row/head/column signs so all output tiles exercise
                # both positive and negative cancellation, with no random seed.
                if role == "x":
                    signs = (1 - 2 * ((torch.arange(shape[0])[:, None] + torch.arange(shape[1])[None, :]) % 2)).to(
                        dtype
                    )
                    for position, value, _ in cancellation_terms(k, kind):
                        data[:, :, position] = signs * value
                else:
                    signs = (1 - 2 * ((torch.arange(shape[0])[:, None] + torch.arange(shape[2])[None, :]) % 2)).to(
                        dtype
                    )
                    for position, _, value in cancellation_terms(k, kind):
                        data[:, position, :] = signs * value
                base = data.to(device)
                view = base
            elif kind == "strided":
                base = random_cpu(tuple(size * 2 + 1 for size in shape), dtype, seed).to(device)
                view = base[tuple(slice(1, size * 2 + 1, 2) for size in shape)]
            elif kind == "strided_dense_suffix":
                base = random_cpu((*shape[:-1], shape[-1] * 2), dtype, seed).to(device)
                view = base[..., ::2]
            elif kind == "transpose_reverse":
                base = random_cpu(tuple(reversed(shape)), dtype, seed).to(device)
                view = base.permute(2, 1, 0)
            elif kind == "transposed":
                base = random_cpu((shape[2], shape[0], shape[1]), dtype, seed).to(device)
                view = base.permute(1, 2, 0)
            elif kind.startswith(role + "_zero_"):
                axis, allocation = int(kind[-1]), list(shape)
                allocation[axis] = 1
                base = random_cpu(tuple(allocation), dtype, seed).to(device)
                view = base.expand(shape)
            else:
                data = random_cpu(shape, dtype, seed)
                if kind in ("special", "nonfinite"):
                    if kind == "special":
                        patterns = {
                            "bf16": (0, 0x8000, 0x3F80, 0xBF80, 0x3F00),
                            "fp16": (0, 0x8000, 0x3C00, 0xBC00, 0x3800),
                            "fp32": (0, 0x80000000, 0x3F800000, 0xBF800000, 0x3F000000),
                        }
                    else:
                        patterns = {
                            "bf16": (0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x3F80),
                            "fp16": (0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x3C00),
                            "fp32": (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x3F800000),
                        }
                    bits = patterns[dtype_name]
                    code = "I" if dtype_name == "fp32" else "H"
                    payload = b"".join(struct.pack("<" + code, bits[i % len(bits)]) for i in range(data.numel()))
                    data = torch.frombuffer(bytearray(payload), dtype=dtype).clone().reshape(shape)
                base = data.to(device)
                if kind == "nz":
                    previous = torch.npu.config.allow_internal_format
                    fixture = {
                        "role": role,
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "allow_internal_format_before": previous,
                    }
                    report.setdefault("nz_fixtures", []).append(fixture)
                    try:
                        torch.npu.config.allow_internal_format = True
                        base = torch_npu.npu_format_cast(base, 29)
                        fixture["actual_format"] = torch_npu.get_npu_format(base)
                        if fixture["actual_format"] != 29:
                            raise RuntimeError("NZ fixture did not construct physical format 29")
                    finally:
                        torch.npu.config.allow_internal_format = previous
                        fixture["allow_internal_format_restored"] = torch.npu.config.allow_internal_format
                        if fixture["allow_internal_format_restored"] != previous:
                            raise RuntimeError("NZ fixture did not restore allow_internal_format")
                view = base
            return base, view

        def oracle(x, weight):
            sequence = report["hf32_sequence"]
            if sequence.get("oracle_phase_complete") and sequence["status"] == "running":
                sequence["candidate_phase_cann_calls"] += 1
                raise RuntimeError("CANN oracle must not run between HF32 candidate sequence checks")
            if args.oracle == "transpose":
                return torch_npu.npu_transpose_batchmatmul(x, weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
            return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)

        def before_call(label, state, expected=None):
            last.clear()
            last.update(
                label=label,
                x=raw(state["x"]),
                weight=raw(state["weight"]),
                x_base=raw(state["bases"][0]),
                weight_base=raw(state["bases"][1]),
            )
            if expected is not None:
                last["expected"] = raw(expected)

        def verify(label, result, expected, state, expected_inputs):
            before_call(label, state, expected)
            last.update(actual=raw(result), x_base_expected=expected_inputs[0], weight_base_expected=expected_inputs[1])
            immutable = all(raw(base) == data for base, data in zip(state["bases"], expected_inputs))
            valid = (
                result.shape == expected.shape == state["shape"]
                and result.dtype == expected.dtype == state["x"].dtype
                and result.device == device
                and result.is_contiguous()
                and torch_npu.get_npu_format(result) == 2
            )
            # A transposed bmm reference need not itself be contiguous.
            if label.endswith("/cann"):
                valid = (
                    result.shape == expected.shape == state["shape"]
                    and result.dtype == expected.dtype
                    and result.device == device
                )
            passed = valid and immutable and last["actual"] == last["expected"]
            report["checks"].append(
                {
                    "case": label,
                    "pass": passed,
                    "input_unchanged": immutable,
                    "actual_sha256": hashlib.sha256(last["actual"]).hexdigest(),
                    "expected_sha256": hashlib.sha256(last["expected"]).hexdigest(),
                    "actual_layout": metadata(result),
                    "reference_layout": state["reference_layout"],
                }
            )
            if not passed:
                raise RuntimeError(f"Strict V-up bit/input/layout gate failed: {label}")

        def hf32_sequence_check():
            """Freeze every CANN oracle before alternating candidate modes/streams."""
            sequence = report["hf32_sequence"]
            sequence.update(
                status="running",
                initial_options=cube_math_snapshot(torch_npu),
                mode_changes=[],
                candidate_options=[],
                frozen_oracles=[],
                candidate_phase_cann_calls=0,
            )
            bank = []

            def set_mode(mode, phase):
                item = {"phase": phase}
                configure_cube_math_type(torch_npu, mode, item)
                sequence["mode_changes"].append(item)

            def checked(label, function):
                last["label"] = label
                before = cube_math_snapshot(torch_npu)
                result = function()
                after = cube_math_snapshot(torch_npu)
                sequence["candidate_options"].append({"case": label, "before": before, "after": after})
                if before != after:
                    raise RuntimeError(f"Candidate changed global math options: {label}")
                return result

            def from_bits(bits, shape):
                count = 1
                for dimension in shape:
                    count *= dimension
                payload = b"".join(struct.pack("<I", bits[index % len(bits)]) for index in range(count))
                return torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(shape)

            try:
                # These explicit fixtures do not change CASES, their order or their seeds.
                for index, (name, dimensions) in enumerate(
                    (
                        ("halfway", (2, 2, 9, 9)),
                        ("tails", (3, 2, 17, 11)),
                        ("nonfinite", (2, 2, 9, 7)),
                    )
                ):
                    n, h, k, v = dimensions
                    if name == "halfway":
                        original_x = from_bits(HF32_ROUNDING_BITS, (n, h, k))
                        original_w = torch.eye(k, v).repeat(h, 1, 1)
                    elif name == "nonfinite":
                        original_x = from_bits(
                            (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x3F800800, 0xBF800800), (n, h, k)
                        )
                        original_w = from_bits(HF32_ROUNDING_BITS, (h, k, v))
                    else:
                        original_x = random_cpu((n, h, k), torch.float32, args.seed + 70000 + index)
                        original_w = random_cpu((h, k, v), torch.float32, args.seed + 71000 + index)
                        original_x.flatten()[: len(HF32_ROUNDING_BITS)].copy_(
                            from_bits(HF32_ROUNDING_BITS, (len(HF32_ROUNDING_BITS),))
                        )
                    variants = {
                        "original": (original_x, original_w),
                        "changed": (original_x.roll(1, -1), original_w.clone()),
                    }
                    x, weight = original_x.to(device), original_w.to(device)
                    state = dict(
                        x=x,
                        weight=weight,
                        bases=(x, weight),
                        shape=(n, h, v),
                        name="hf32-sequence/" + name,
                        variants=variants,
                        oracles={},
                        graphs={},
                        outputs={},
                    )
                    for stage, inputs in variants.items():
                        for base, data in zip(state["bases"], inputs):
                            base.copy_(data)
                        expected_inputs = tuple(raw(data) for data in inputs)
                        for mode in (0, 3):
                            set_mode(mode, f"freeze/{name}/{stage}/{mode}")
                            before_call(f"{state['name']}/freeze/{stage}/{mode}", state)
                            native = oracle(x, weight)
                            expected = native.cpu()
                            if any(raw(base) != data for base, data in zip(state["bases"], expected_inputs)):
                                raise RuntimeError("CANN changed inputs while freezing HF32 sequence oracle")
                            state["oracles"][stage, mode] = expected, metadata(native)
                            sequence["frozen_oracles"].append(
                                {
                                    "case": name,
                                    "stage": stage,
                                    "mode": mode,
                                    "input_sha256": [hashlib.sha256(data).hexdigest() for data in expected_inputs],
                                    "output_sha256": hashlib.sha256(raw(expected)).hexdigest(),
                                }
                            )
                    if name == "halfway" and raw(state["oracles"]["original", 0][0]) == raw(
                        state["oracles"]["original", 3][0]
                    ):
                        raise RuntimeError("HF32 halfway fixture did not distinguish native modes 0 and 3")
                    bank.append(state)
                sequence["oracle_phase_complete"] = True
                save()
                # No CANN invocation is allowed below this point, including warmup/capture.
                streams = (torch.npu.Stream(), torch.npu.Stream())
                for state in bank:
                    for base, data in zip(state["bases"], state["variants"]["original"]):
                        base.copy_(data)
                    for mode in (0, 3):
                        set_mode(mode, f"capture/{state['name']}/{mode}")
                        run = lambda state=state: module.vup_bmm(state["x"], state["weight"])
                        checked(f"{state['name']}/warmup/{mode}", run)
                        torch.npu.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            state["outputs"][mode] = checked(f"{state['name']}/capture/{mode}", run)
                        state["graphs"][mode] = graph
                    keep_alive.append(state)
                    for stage in ("original", "changed", "restored"):
                        variant = "changed" if stage == "changed" else "original"
                        inputs = state["variants"][variant]
                        for base, data in zip(state["bases"], inputs):
                            base.copy_(data)
                            if raw(base) != raw(data):
                                raise RuntimeError("HF32 sequence live input copy changed bits")
                        expected_inputs = tuple(raw(data) for data in inputs)
                        torch.npu.synchronize()

                        def invoke(mode, scope, state=state):
                            if scope == "graph":
                                state["graphs"][mode].replay()
                                return state["outputs"][mode]
                            return module.vup_bmm(state["x"], state["weight"])

                        def check_output(
                            label, mode, result, state=state, variant=variant, expected_inputs=expected_inputs
                        ):
                            expected, layout = state["oracles"][variant, mode]
                            state["reference_layout"] = layout
                            verify(label, result, expected, state, expected_inputs)

                        for scope in ("eager", "graph"):
                            for repeat in range(args.graph_repeats):
                                for mode in (3, 0, 3):
                                    label = f"{state['name']}/{stage}/{scope}/{repeat}/{mode}"
                                    set_mode(mode, label)
                                    result = checked(label, lambda mode=mode, scope=scope: invoke(mode, scope))
                                    torch.npu.synchronize()
                                    check_output(label, mode, result)
                            for modes in ((3, 0), (0, 3)):
                                results = []
                                if scope == "graph":
                                    # Captured graphs retain their own mode. Avoid a setter between
                                    # stream launches: a setter may synchronize and hide leakage.
                                    set_mode(0, f"{state['name']}/{stage}/graph/streams/{modes}")
                                for slot, (stream, mode) in enumerate(zip(streams, modes)):
                                    label = f"{state['name']}/{stage}/{scope}/streams/{modes}/{slot}/{mode}"
                                    if scope == "eager":
                                        set_mode(mode, label)
                                    with torch.npu.stream(stream):
                                        result = checked(label, lambda mode=mode, scope=scope: invoke(mode, scope))
                                    results.append((label, mode, result))
                                for stream in streams:
                                    stream.synchronize()
                                for label, mode, result in results:
                                    check_output(label, mode, result)
                        save()
                sequence["status"] = "passed"
            finally:
                # Public setter requires an enum, so the CLI requires an explicit initial mode.
                set_mode(args.cube_math_type, "restore")
                sequence["final_options"] = cube_math_snapshot(torch_npu)
                sequence["options_restored"] = sequence["final_options"] == sequence["initial_options"]
                if not sequence["options_restored"]:
                    sequence["status"] = "failed"
                    raise RuntimeError("HF32 sequence did not restore its explicit initial math options")
                if sequence["status"] != "passed":
                    sequence["status"] = "failed"
                save()

        with torch.inference_mode():
            for dtype_name in report["selected_dtypes"]:
                # Preserve all 16-bit payloads, including NaNs and signed zero,
                # independently of BMM arithmetic which can hide copy errors.
                for layout in ("stride2", "transpose"):
                    step = 2 if layout == "stride2" else 1
                    shape = (4, 4, 4096 * step)
                    payload = copy_bit_payload(dtype_name, math.prod(shape), step)
                    backing = (
                        torch.frombuffer(bytearray(payload), dtype=dtypes[dtype_name]).clone().reshape(shape).to(device)
                    )
                    view = backing[..., ::2] if layout == "stride2" else backing.permute(2, 1, 0)
                    expected_bytes, backing_before = raw(view), raw(backing)
                    copied = module._contiguous_copy(view)
                    actual_bytes = raw(copied)
                    passed = (
                        copied.shape == view.shape
                        and copied.dtype == view.dtype
                        and copied.is_contiguous()
                        and actual_bytes == expected_bytes
                        and raw(backing) == backing_before
                    )
                    report["copy_bit_checks"].append(
                        {
                            "case": dtype_name + "/raw_copy/" + layout,
                            "pass": passed,
                            "distinct_logical_patterns": 8 if dtype_name == "fp32" else 65536,
                            "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
                            "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                            "input": metadata(view),
                            "output": metadata(copied),
                        }
                    )
                    if not passed:
                        last.update(
                            label=dtype_name + "/raw_copy/" + layout,
                            actual=actual_bytes,
                            expected=expected_bytes,
                            x_base=raw(backing),
                        )
                        raise RuntimeError("Raw bit-copy changed payload, layout or input storage")
                    save()
                for case_index, (name, dimensions, kind) in enumerate(selected):
                    n, h, k, v = dimensions
                    x_base, x = operand((n, h, k), kind, "x", dtype_name, args.seed + case_index)
                    w_base, weight = operand((h, k, v), kind, "weight", dtype_name, args.seed + case_index + 1000)
                    bases = x_base, w_base
                    original = tuple(base.cpu().clone() for base in bases)
                    state = {
                        "x": x,
                        "weight": weight,
                        "bases": bases,
                        "shape": (n, h, v),
                        "name": f"{dtype_name}/{name}",
                    }
                    state["input_layouts"] = [metadata(x), metadata(weight)]
                    report["cases"].append(
                        {
                            "case": state["name"],
                            "input_layouts": state["input_layouts"],
                            "base_layouts": [metadata(base) for base in bases],
                        }
                    )
                    if kind.startswith("cancel_"):
                        fixture = {
                            "kind": kind,
                            "terms": cancellation_terms(k, kind),
                            "exact_real_sum_before_sign": 1.75,
                            "output_oracle": "CANN bits, not the exact real sum",
                        }
                        report["cases"][-1]["cancellation_fixture"] = fixture
                        if dtype_name in ("bf16", "fp16"):
                            block_m, block_k, block_v = module._half_tiling(
                                n, h, k, v, x.element_size(), module.get_device_resources(args.device)
                            )
                            fixture["candidate_tiles_mkv"] = [block_m, block_k, block_v]
                            if not (k > block_k and v > block_v and (name != "cancel_mkv" or n > block_m)):
                                raise RuntimeError(
                                    "Cancellation fixture no longer crosses its required candidate tiles"
                                )
                    before_call(state["name"] + "/eager", state)
                    expected_inputs = tuple(raw(base) for base in bases)
                    native = oracle(x, weight)
                    state["reference_layout"] = metadata(native)
                    expected = native.cpu()
                    last["expected"] = raw(expected)
                    verify(state["name"] + "/eager/triton", module.vup_bmm(x, weight), expected, state, expected_inputs)
                    if n * h * v == 0:
                        continue
                    graphs, outputs = {}, {}
                    for mode in ("cann", "triton"):
                        before_call(state["name"] + "/capture/" + mode, state, expected)
                        run = oracle if mode == "cann" else module.vup_bmm
                        run(x, weight)
                        torch.npu.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            outputs[mode] = run(x, weight)
                        graphs[mode] = graph
                    state.update(graphs=graphs, outputs=outputs, original=original)
                    keep_alive.append(state)
                    original_expected = None
                    for stage in ("original", "changed", "restored"):
                        for index, base in enumerate(bases):
                            data = (
                                random_cpu(base.shape, base.dtype, args.seed + case_index + index + 20000)
                                if stage == "changed"
                                else original[index]
                            )
                            base.copy_(data)
                            if raw(base) != raw(data):
                                raise RuntimeError("Live graph base input copy changed bits")
                        expected_inputs = tuple(raw(base) for base in bases)
                        before_call(state["name"] + "/graph/" + stage, state)
                        native = oracle(x, weight)
                        state["reference_layout"] = metadata(native)
                        expected = native.cpu()
                        last["expected"] = raw(expected)
                        if stage == "original":
                            original_expected = raw(expected)
                        if stage == "changed" and k and kind != "nonfinite" and raw(expected) == original_expected:
                            raise RuntimeError("Changed live inputs did not change the current CANN oracle")
                        for repeat in range(args.graph_repeats):
                            for mode in ("cann", "triton"):
                                graphs[mode].replay()
                                torch.npu.synchronize()
                                verify(
                                    f"{state['name']}/graph/{stage}/{repeat}/{mode}",
                                    outputs[mode],
                                    expected,
                                    state,
                                    expected_inputs,
                                )
                    state.update(expected=expected, expected_inputs=expected_inputs)
                    if name in (args.timing_case or ("legacy", "mtp_n1")):
                        timing_jobs.append(state)
                    save()
            if args.hf32_sequence_check:
                hf32_sequence_check()
            if (
                (not args.raw_copy_only and not report["checks"])
                or not all(item["pass"] for item in report["checks"])
                or not report["copy_bit_checks"]
                or not all(item["pass"] for item in report["copy_bit_checks"])
            ):
                raise RuntimeError("Missing or failed numerical checks; timing forbidden")
            report.update(
                numerical_checks_passed=True,
                npu_domains_pending=args.raw_copy_only,
                retained_graph_count=sum(len(state["graphs"]) for state in keep_alive),
            )
            save()
            if args.timing:
                if not timing_jobs:
                    raise ValueError("No nonempty selected --timing-case is available")
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                end.record()
                end.synchronize()
                for state in timing_jobs:
                    for scope in ("eager", "graph"):
                        before_call(state["name"] + "/timing/" + scope, state, state["expected"])

                        def invoke(mode, scope=scope, state=state):
                            if scope == "graph":
                                state["graphs"][mode].replay()
                                return state["outputs"][mode]
                            return (oracle if mode == "cann" else module.vup_bmm)(state["x"], state["weight"])

                        for mode in ("cann", "triton"):
                            for _ in range(5):
                                invoke(mode)
                        torch.npu.synchronize()
                        final_outputs = {}
                        for block in range(args.samples):
                            for position, mode in enumerate(("cann", "triton", "triton", "cann")):
                                before = time.perf_counter_ns()
                                start.record()
                                final_outputs[mode] = invoke(mode)
                                end.record()
                                submitted = time.perf_counter_ns()
                                end.synchronize()
                                finished = time.perf_counter_ns()
                                report["timings"].append(
                                    {
                                        "case": state["name"],
                                        "scope": scope,
                                        "mode": mode,
                                        "block": block,
                                        "position": position,
                                        "device_us": start.elapsed_time(end) * 1000,
                                        "host_submit_us": (submitted - before) / 1000,
                                        "full_wall_us": (finished - before) / 1000,
                                    }
                                )
                        # Check ACTUAL timed outputs before another replay/copy/oracle/warmup.
                        for mode in ("cann", "triton"):
                            verify(
                                f"{state['name']}/timed/{scope}/{mode}",
                                final_outputs[mode],
                                state["expected"],
                                state,
                                state["expected_inputs"],
                            )
                        save()
                report["summary"] = []
                for state in timing_jobs:
                    for scope in ("eager", "graph"):
                        values = {
                            mode: {
                                metric: statistics.mean(
                                    t[metric]
                                    for t in report["timings"]
                                    if t["case"] == state["name"] and t["scope"] == scope and t["mode"] == mode
                                )
                                for metric in ("device_us", "host_submit_us", "full_wall_us")
                            }
                            for mode in ("cann", "triton")
                        }
                        report["summary"].append(
                            {
                                "case": state["name"],
                                "scope": scope,
                                "mean_us": values,
                                "saving_us": {
                                    metric: values["cann"][metric] - values["triton"][metric]
                                    for metric in values["cann"]
                                },
                            }
                        )
                report["performance_results_valid"] = True
            verify_control("end")
            verify_resources("end", module)
            report.update(status="passed", phase="complete")
            save()
    except Exception as error:
        failure = output / "first-failure"
        failure.mkdir(exist_ok=True)
        for key, data in last.items():
            if isinstance(data, bytes):
                (failure / f"{key}.bin").write_bytes(data)
        report.update(
            status="failed",
            numerical_checks_passed=False,
            performance_results_valid=False,
            error=repr(error),
            traceback=traceback.format_exc(),
            failure_case=last.get("label"),
        )
        save()
        raise
    finally:
        if not report["control_artifact_checks"] or report["control_artifact_checks"][-1]["phase"] != "end":
            try:
                verify_control("end")
            except Exception as error:
                report["control_artifacts_end_error"] = repr(error)
            save()


if __name__ == "__main__":
    main()
