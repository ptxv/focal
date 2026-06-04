import argparse
import csv
import math
import os
import statistics
import sys

import torch

import focal
from focal.ops import _decode_attn_contig_out, cuda_extension_available

MAX_ABS_TOL = 5e-2
RMS_TOL = 5e-3


CSV_FIELDS = [
    "backend",
    "device_name",
    "torch_version",
    "torch_cuda_version",
    "dtype",
    "B",
    "Hq",
    "Hkv",
    "D",
    "L",
    "warmup",
    "repeats",
    "p50_us",
    "p90_us",
    "mean_us",
    "min_us",
    "max_us",
    "max_abs_error",
    "rms_error",
]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", choices=["ours", "torch_ref"], default="ours")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--L", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--csv")
    return parser.parse_args()


def dtype_from_name(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def validate_args(args):
    if args.B <= 0:
        raise SystemExit("--B must be positive")
    if args.Hq <= 0:
        raise SystemExit("--Hq must be positive")
    if args.Hkv <= 0:
        raise SystemExit("--Hkv must be positive")
    if args.Hq % args.Hkv != 0:
        raise SystemExit("--Hq must be divisible by --Hkv")
    if args.D <= 0:
        raise SystemExit("--D must be positive")
    if args.L <= 0:
        raise SystemExit("--L must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.backend == "ours" and args.D != 128:
        raise SystemExit("backend=ours requires --D 128 for CUDA kernel v1")
    if args.backend == "ours" and args.dtype != "fp16":
        raise SystemExit("backend=ours requires --dtype fp16 for CUDA kernel v1")


def make_inputs(args, dtype):
    q = torch.randn((args.B, args.Hq, args.D), device="cuda", dtype=dtype)
    k = torch.randn((args.B, args.Hkv, args.L, args.D), device="cuda", dtype=dtype)
    v = torch.randn((args.B, args.Hkv, args.L, args.D), device="cuda", dtype=dtype)
    seq_lens = torch.full((args.B,), args.L, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    return q, k, v, seq_lens, out


def make_ours_runner(q, k, v, seq_lens, out, sm_scale):
    def run():
        # The public op is checked before timing; this path only avoids output allocation.
        _decode_attn_contig_out(q, k, v, seq_lens, out, sm_scale)

    return run


def make_torch_ref_runner(args, q, k, v, out, sm_scale):
    group_size = args.Hq // args.Hkv
    hkv_indices = torch.arange(args.Hq, device="cuda", dtype=torch.int64) // group_size

    q_f = q.float().contiguous().view(args.B * args.Hq, args.D, 1)
    k_f = k.index_select(1, hkv_indices).float().contiguous().view(args.B * args.Hq, args.L, args.D)
    v_f = v.index_select(1, hkv_indices).float().contiguous().view(args.B * args.Hq, args.L, args.D)

    scores = torch.empty((args.B * args.Hq, args.L, 1), device="cuda", dtype=torch.float32)
    scores_2d = scores.view(args.B * args.Hq, args.L)
    max_buf = torch.empty((args.B * args.Hq, 1), device="cuda", dtype=torch.float32)
    denom = torch.empty((args.B * args.Hq, 1), device="cuda", dtype=torch.float32)
    probs = torch.empty((args.B * args.Hq, args.L), device="cuda", dtype=torch.float32)
    probs_bmm = probs.view(args.B * args.Hq, 1, args.L)
    out_f = torch.empty((args.B * args.Hq, 1, args.D), device="cuda", dtype=torch.float32)
    out_view = out_f.view(args.B, args.Hq, args.D)

    def run():
        torch.bmm(k_f, q_f, out=scores)
        scores_2d.mul_(sm_scale)
        torch.amax(scores_2d, dim=1, keepdim=True, out=max_buf)
        torch.sub(scores_2d, max_buf, out=probs)
        torch.exp(probs, out=probs)
        torch.sum(probs, dim=1, keepdim=True, out=denom)
        probs.div_(denom)
        torch.bmm(probs_bmm, v_f, out=out_f)
        out.copy_(out_view)

    return run


def error_metrics(got, expected):
    diff = got.float() - expected.float()
    abs_diff = diff.abs()
    return abs_diff.max().item(), torch.sqrt(torch.mean(diff * diff)).item()


def correctness_check(args, run, q, k, v, seq_lens, out, sm_scale):
    if args.L > 2048:
        raise SystemExit(
            "Correctness check is required before timing; this benchmark currently checks L <= 2048."
        )

    expected = focal.ref_decode_attn_contig(q, k, v, seq_lens, sm_scale)
    if args.backend == "ours":
        got = focal.decode_attn_contig(q, k, v, seq_lens, sm_scale)
    else:
        run()
        got = out
    torch.cuda.synchronize()
    return error_metrics(got, expected)


def enforce_correctness(max_abs_error, rms_error):
    if max_abs_error > MAX_ABS_TOL or rms_error > RMS_TOL:
        raise SystemExit(
            "correctness check failed before timing: "
            f"max_abs_error={max_abs_error:.6g}, rms_error={rms_error:.6g}"
        )


def time_runner(run, warmup, repeats):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        starts[i].record()
        run()
        ends[i].record()

    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(repeats)]


def percentile(sorted_values, p):
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarize_times(times):
    ordered = sorted(times)
    return {
        "p50_us": percentile(ordered, 0.50),
        "p90_us": percentile(ordered, 0.90),
        "mean_us": statistics.fmean(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def write_csv(path, row):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_table(row):
    headers = ["backend", "dtype", "B", "Hq", "Hkv", "D", "L", "p50_us", "p90_us", "mean_us"]
    values = [
        row["backend"],
        row["dtype"],
        row["B"],
        row["Hq"],
        row["Hkv"],
        row["D"],
        row["L"],
        f"{row['p50_us']:.3f}",
        f"{row['p90_us']:.3f}",
        f"{row['mean_us']:.3f}",
    ]
    widths = [max(len(str(h)), len(str(v))) for h, v in zip(headers, values)]
    print("  ".join(str(h).rjust(w) for h, w in zip(headers, widths)))
    print("  ".join(str(v).rjust(w) for v, w in zip(values, widths)))


def main():
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this benchmark on an NVIDIA GPU host.")
    if args.backend == "ours" and not cuda_extension_available():
        raise SystemExit(
            "focal CUDA extension is unavailable. Build it with: "
            "FOCAL_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation"
        )

    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(0)
    q, k, v, seq_lens, out = make_inputs(args, dtype)
    sm_scale = 1.0 / math.sqrt(args.D)

    if args.backend == "ours":
        run = make_ours_runner(q, k, v, seq_lens, out, sm_scale)
    else:
        run = make_torch_ref_runner(args, q, k, v, out, sm_scale)

    max_abs_error, rms_error = correctness_check(args, run, q, k, v, seq_lens, out, sm_scale)
    enforce_correctness(max_abs_error, rms_error)
    times = time_runner(run, args.warmup, args.repeats)
    stats = summarize_times(times)

    row = {
        "backend": args.backend,
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "B": args.B,
        "Hq": args.Hq,
        "Hkv": args.Hkv,
        "D": args.D,
        "L": args.L,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_abs_error": max_abs_error,
        "rms_error": rms_error,
        **stats,
    }

    print_table(row)
    if args.csv:
        write_csv(args.csv, row)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
