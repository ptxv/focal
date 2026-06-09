import argparse
import json
from pathlib import Path

import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03
SHAPES = [
    (M, K, N)
    for M in focal_w4a16.SUPPORTED_M
    for K in focal_w4a16.SUPPORTED_K
    for N in focal_w4a16.SUPPORTED_N
] + list(focal_w4a16.SUPPORTED_EXTRA_SHAPES)


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


def baseline_dequant_matmul(x, wq, scales, zeros, K):
    # Baseline includes dequantization every iteration.
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
    return (x.float() @ w_deq.T).to(torch.bfloat16)


def baseline_predequant_matmul(x, w_deq):
    # Predequant baseline isolates matmul cost.
    return (x.float() @ w_deq.T).to(torch.bfloat16)


def default_jsonl(gpu_name):
    safe_name = "".join(c if c.isalnum() else "_" for c in gpu_name).strip("_")
    return Path("bench") / f"results_{safe_name}.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--jsonl", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    gpu_name = torch.cuda.get_device_name()
    jsonl_path = Path(args.jsonl) if args.jsonl else default_jsonl(gpu_name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'gpu_name':28} {'M':>2} {'K':>5} {'N':>5} "
        f"{'focal_ms':>10} {'deq_mm_ms':>10} {'predeq_mm_ms':>13} "
        f"{'spd_deq':>8} {'spd_pre':>8} {'abs_err':>9} {'rel_err':>9}"
    )
    print(header)
    print("-" * len(header))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for M, K, N in SHAPES:
                    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=1234)

                    y = focal_w4a16.w4a16_linear(x, wq, scales, zeros)
                    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
                    ref = baseline_predequant_matmul(x, w_deq)
                    torch.cuda.synchronize()
                    max_abs_err, max_rel_err = max_errors(y, ref)

                    focal_ms = time_cuda(
                        lambda: focal_w4a16.w4a16_linear(x, wq, scales, zeros),
                        args.warmup,
                        args.iters,
                    )
                    baseline_dequant_matmul_ms = time_cuda(
                        lambda: baseline_dequant_matmul(x, wq, scales, zeros, K),
                        args.warmup,
                        args.iters,
                    )
                    baseline_predequant_matmul_ms = time_cuda(
                        lambda: baseline_predequant_matmul(x, w_deq),
                        args.warmup,
                        args.iters,
                    )

                    result = {
                        "gpu_name": gpu_name,
                        "M": M,
                        "K": K,
                        "N": N,
                        "focal_ms": focal_ms,
                        "baseline_dequant_matmul_ms": baseline_dequant_matmul_ms,
                        "baseline_predequant_matmul_ms": baseline_predequant_matmul_ms,
                        "speedup_vs_dequant_matmul": baseline_dequant_matmul_ms / focal_ms,
                        "speedup_vs_predequant_matmul": baseline_predequant_matmul_ms / focal_ms,
                        "max_abs_err": max_abs_err,
                        "max_rel_err": max_rel_err,
                        "max_abs_err_threshold": MAX_ABS_ERR,
                        "max_rel_err_threshold": MAX_REL_ERR,
                    }
                    # JSONL preserves exact benchmark rows.
                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    print(
                        f"{gpu_name[:28]:28} {M:2d} {K:5d} {N:5d} "
                        f"{focal_ms:10.4f} {baseline_dequant_matmul_ms:10.4f} "
                        f"{baseline_predequant_matmul_ms:13.4f} "
                        f"{result['speedup_vs_dequant_matmul']:8.3f} "
                        f"{result['speedup_vs_predequant_matmul']:8.3f} "
                        f"{max_abs_err:9.6f} {max_rel_err:9.6f}"
                    )

    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
