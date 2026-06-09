import argparse
import json
import statistics
from pathlib import Path

import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03
SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (8, 8192, 11008),
]
LAYERS = [
    ("q_proj", 1, 4096, 4096),
    ("k_proj", 1, 4096, 4096),
    ("v_proj", 1, 4096, 4096),
    ("o_proj", 1, 4096, 4096),
    ("gate_proj", 1, 4096, 11008),
    ("up_proj", 1, 4096, 11008),
    ("down_proj", 1, 11008, 4096),
]


def max_errors(y, ref):
    diff = (y.float() - ref.float()).abs()
    max_abs_err = diff.max()
    max_rel_err = (diff / ref.float().abs().clamp_min(1.0)).max()
    return max_abs_err.item(), max_rel_err.item()


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def summarize(samples):
    ordered = sorted(samples)
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def dequant_matmul(x, wq, scales, zeros, K):
    # This baseline pays dequant cost every call.
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
    return (x.float() @ w_deq.T).to(torch.bfloat16)


def predequant_matmul(x, w_deq):
    # This baseline keeps only matmul cost.
    return (x.float() @ w_deq.T).to(torch.bfloat16)


def bench_shape(M, K, N, seed, warmup, iters, repeats):
    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=seed)
    y = focal_w4a16.w4a16_linear(x, wq, scales, zeros)
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
    ref = predequant_matmul(x, w_deq)
    torch.cuda.synchronize()
    max_abs_err, max_rel_err = max_errors(y, ref)

    focal = []
    dequant = []
    predequant = []
    for _ in range(repeats):
        focal.append(
            time_cuda(lambda: focal_w4a16.w4a16_linear(x, wq, scales, zeros), warmup, iters)
        )
        dequant.append(
            time_cuda(lambda: dequant_matmul(x, wq, scales, zeros, K), warmup, iters)
        )
        predequant.append(
            time_cuda(lambda: predequant_matmul(x, w_deq), warmup, iters)
        )

    focal_summary = summarize(focal)
    dequant_summary = summarize(dequant)
    predequant_summary = summarize(predequant)
    return {
        "M": M,
        "K": K,
        "N": N,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "focal": focal_summary,
        "dequant_matmul": dequant_summary,
        "predequant_matmul": predequant_summary,
        "median_speedup_vs_dequant": (
            dequant_summary["median_ms"] / focal_summary["median_ms"]
        ),
        "median_speedup_vs_predequant": (
            predequant_summary["median_ms"] / focal_summary["median_ms"]
        ),
    }


def make_case(M, K, N, seed):
    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=seed)
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
    return x, wq, scales, zeros, w_deq


def bench_bundle(warmup, iters, repeats):
    cases = []
    for index, (name, M, K, N) in enumerate(LAYERS):
        cases.append((name, K, make_case(M, K, N, seed=9000 + index)))

    def focal_bundle():
        return [
            focal_w4a16.w4a16_linear(x, wq, scales, zeros)
            for _, _, (x, wq, scales, zeros, _) in cases
        ]

    def dequant_bundle():
        return [
            dequant_matmul(x, wq, scales, zeros, K)
            for _, K, (x, wq, scales, zeros, _) in cases
        ]

    def predequant_bundle():
        return [predequant_matmul(x, w_deq) for _, _, (x, _, _, _, w_deq) in cases]

    max_abs_err = 0.0
    max_rel_err = 0.0
    for _, _, (x, wq, scales, zeros, w_deq) in cases:
        y = focal_w4a16.w4a16_linear(x, wq, scales, zeros)
        ref = predequant_matmul(x, w_deq)
        torch.cuda.synchronize()
        abs_err, rel_err = max_errors(y, ref)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)

    focal = []
    dequant = []
    predequant = []
    for _ in range(repeats):
        focal.append(time_cuda(focal_bundle, warmup, iters))
        dequant.append(time_cuda(dequant_bundle, warmup, iters))
        predequant.append(time_cuda(predequant_bundle, warmup, iters))

    focal_summary = summarize(focal)
    dequant_summary = summarize(dequant)
    predequant_summary = summarize(predequant)
    return {
        "name": "llama2_7b_supported_decode_projection_bundle",
        "layers": [
            {"name": name, "M": M, "K": K, "N": N}
            for name, M, K, N in LAYERS
        ],
        "unsupported_layers": [],
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "focal": focal_summary,
        "dequant_matmul": dequant_summary,
        "predequant_matmul": predequant_summary,
        "median_speedup_vs_dequant": (
            dequant_summary["median_ms"] / focal_summary["median_ms"]
        ),
        "median_speedup_vs_predequant": (
            predequant_summary["median_ms"] / focal_summary["median_ms"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", type=str, default="bench/decode_bundle.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    out = {
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "shape_distributions": [
            bench_shape(M, K, N, 7000 + i, args.warmup, args.iters, args.repeats)
            for i, (M, K, N) in enumerate(SHAPES)
        ],
        "decode_bundle": bench_bundle(args.warmup, args.iters, args.repeats),
        "notes": [
            "Bundle covers supported Llama-2-7B decode projections.",
            "Bundle covers all listed Llama-2-7B projection shapes.",
            "Third-party kernels are not compared in this script.",
        ],
    }

    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
