# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU oracle checks for quantize + DP pack; runnable without pytest.

python tests/ut/ops/test_oproj_quant_pack.py --probe-only --output probe.json
python tests/ut/ops/test_oproj_quant_pack.py --output checks.json
Optional --parameter-json accepts [{"name": ..., "scale_recip": ..., "offset": ...}].
Values are the stored BF16 reciprocal/offset, never a newly computed reciprocal.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch_npu


def _parameters(path=None):
    cases = [
        {"name": "unit", "scale_recip": 1.0, "offset": 0.0},
        {"name": "half_tie", "scale_recip": 1.0, "offset": 0.5},
        {"name": "non_power_large", "scale_recip": 3.140625, "offset": 1.25},
        {"name": "non_power_small", "scale_recip": 0.333984375, "offset": -3.25},
        {"name": "small", "scale_recip": 0.0078125, "offset": 3.0},
        {"name": "large", "scale_recip": 128.0, "offset": -2.5},
    ]
    if path:
        extra = json.loads(Path(path).read_text())
        if not isinstance(extra, list):
            raise ValueError("parameter-json must contain a list of stored scalar reciprocal/offset pairs")
        cases.extend(extra)
    for case in cases:
        if any(not math.isfinite(float(case[key])) for key in ("scale_recip", "offset")):
            raise ValueError("Test parameters must be finite")
    return cases


def _finite_bf16():
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    return values[torch.isfinite(values)]


def _oracle(x, reciprocal, offset):
    width = x.shape[-1]
    scale = reciprocal.expand(width).contiguous() if reciprocal.numel() == 1 else reciprocal
    zero = offset.expand(width).contiguous() if offset.numel() == 1 else offset
    return torch_npu.npu_quantize(x, scale, zero, torch.qint8, -1, False)


def probe_quantize(cases, device):
    finite = _finite_bf16()
    width = 8192
    flat = torch.zeros(math.ceil(finite.numel() / width) * width, dtype=torch.bfloat16)
    flat[:finite.numel()] = finite
    x = flat.reshape(-1, width).to(device)
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0,
                            torch.finfo(torch.bfloat16).tiny / 128,
                            -torch.finfo(torch.bfloat16).tiny / 128], dtype=torch.bfloat16).reshape(1, -1).to(device)
    records = []
    for case in cases:
        scale = torch.tensor([case["scale_recip"]], dtype=torch.bfloat16)
        zero = torch.tensor([case["offset"]], dtype=torch.bfloat16)
        expected = _oracle(x, scale.to(device), zero.to(device)).cpu().flatten()[:finite.numel()]
        fp_x, fp_scale, fp_zero = finite.float(), scale.float(), zero.float()
        product = fp_x * fp_scale
        variants = {
            "fp32_mul_add_separate": product + fp_zero,
            "bf16_mul_fp32_add": product.bfloat16().float() + fp_zero,
            "bf16_mul_bf16_add": (product.bfloat16().float() + fp_zero).bfloat16().float(),
            "fp32_mul_add_bf16": (product + fp_zero).bfloat16().float(),
            "fp32_mul_add_fp16": (product + fp_zero).half().float(),
        }
        counts = {}
        examples = {}
        for name, affine in variants.items():
            candidate = torch.round(affine).clamp(-128, 127).to(torch.int8)
            mismatch = torch.nonzero(candidate != expected).flatten()
            counts[name] = mismatch.numel()
            examples[name] = [{"x": float(finite[i]), "oracle": int(expected[i]),
                               "candidate": int(candidate[i])} for i in mismatch[:6]]
        mapping = _oracle(special, scale.to(device), zero.to(device)).cpu().flatten().tolist()
        records.append({"name": case["name"], "stored_scale_recip": scale.item(),
                        "stored_offset": zero.item(), "finite_bf16_patterns": finite.numel(),
                        "mismatch_counts": counts, "first_mismatches": examples,
                        "special_input_order": ["nan", "+inf", "-inf", "+0", "-0", "+min_subnormal", "-min_subnormal"],
                        "special_oracle_int8": mapping})
    return records


def check_quant_pack(cases, device, kernel_path=None, block_size=1024):
    # Delayed import keeps the oracle probe runnable before a kernel is installed.
    if kernel_path:
        spec = importlib.util.spec_from_file_location("oproj_quant_pack_under_test", kernel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        quantize_pack = module.quantize_pack
        prepare_parameters = module.prepare_parameters
    else:
        from vllm_ascend.ops.triton.oproj_quant_pack import prepare_parameters, quantize_pack

    records = []

    def compare(label, x, scale, offset, graph=False):
        expected = _oracle(x, scale, offset).reshape(x.shape[0], 2, x.shape[1] // 2).transpose(0, 1).contiguous()
        prepared_scale, prepared_offset = prepare_parameters(scale, offset)
        actual = quantize_pack(x, prepared_scale, prepared_offset, exchange_size=2, block_size=block_size)
        if graph:
            quantize_pack(x, prepared_scale, prepared_offset, exchange_size=2, block_size=block_size)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                actual = quantize_pack(x, prepared_scale, prepared_offset, exchange_size=2, block_size=block_size)
            # Verify a replay consumes the live input, including changed values.
            x.add_(0.125)
            for _ in range(3):
                captured.replay()
            expected = _oracle(x, scale, offset).reshape(x.shape[0], 2, x.shape[1] // 2).transpose(0, 1).contiguous()
        torch.npu.synchronize()
        mismatch = int(torch.count_nonzero(actual != expected).item())
        records.append({"case": label, "shape": list(x.shape), "graph": graph,
                        "mismatches": mismatch, "pass": mismatch == 0 and actual.is_contiguous()
                        and actual.dtype == torch.int8 and actual.shape == expected.shape})

    torch.manual_seed(20261024)
    for n in (1, 6, 16, 64):
        for case in cases:
            scale = torch.tensor([case["scale_recip"]], device=device, dtype=torch.bfloat16)
            zero = torch.tensor([case["offset"]], device=device, dtype=torch.bfloat16)
            x = (torch.randn((n, 8192)) * 48).to(device=device, dtype=torch.bfloat16)
            compare(f"random_scalar_{case['name']}", x, scale, zero)
            compare(f"random_repeated_{case['name']}", x, scale.expand(8192).contiguous(), zero.expand(8192).contiguous())
        scale = torch.tensor([0.333984375, 3.140625, 8.0], dtype=torch.bfloat16).repeat(2731)[:8192].to(device)
        zero = torch.tensor([-3.25, 0.5, 1.25], dtype=torch.bfloat16).repeat(2731)[:8192].to(device)
        compare("per_channel", x, scale, zero)
        compare("per_channel_replay", x, scale, zero, graph=True)
        edge = torch.arange(-260, 261).float() / 2
        x = edge.repeat(math.ceil(n * 8192 / edge.numel()))[:n * 8192].reshape(n, 8192).to(device=device, dtype=torch.bfloat16)
        compare("half_ties_saturation", x, torch.ones(1, device=device, dtype=torch.bfloat16),
                torch.zeros(1, device=device, dtype=torch.bfloat16))

    finite = _finite_bf16()
    x = torch.zeros((8, 8192), dtype=torch.bfloat16)
    x.flatten()[:finite.numel()] = finite
    x = x.to(device)
    for case in cases:
        scale = torch.tensor([case["scale_recip"]], device=device, dtype=torch.bfloat16)
        zero = torch.tensor([case["offset"]], device=device, dtype=torch.bfloat16)
        compare(f"all_finite_bf16_{case['name']}", x, scale, zero)
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0], dtype=torch.bfloat16).repeat(2048).reshape(1, 8192).to(device)
    compare("special_values", special, torch.ones(1, device=device, dtype=torch.bfloat16),
            torch.zeros(1, device=device, dtype=torch.bfloat16))
    return records


def test_oproj_quant_pack():
    records = check_quant_pack(_parameters(), torch.device("npu"))
    assert all(item["pass"] for item in records), json.dumps(records, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--parameter-json")
    parser.add_argument("--quant-pack-path", help="Load the candidate file without installing it into the service")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 32 <= args.block_size <= 4096 or args.block_size & (args.block_size - 1):
        parser.error("block-size must be a power of two from 32 to 4096")
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    result = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "device": str(device), "quant_pack_path": args.quant_pack_path,
              "quant_pack_block_size": args.block_size, "checks": []}
    try:
        cases = _parameters(args.parameter_json)
        result["probe"] = probe_quantize(cases, device)
        if not args.probe_only:
            result["checks"] = check_quant_pack(cases, device, args.quant_pack_path, args.block_size)
        result["status"] = "probe_only" if args.probe_only else ("pass" if all(item["pass"] for item in result["checks"]) else "fail")
    except Exception as exc:
        result["status"] = "error"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]), "output": args.output}))
    raise SystemExit(1 if result["status"] == "fail" else 0)
