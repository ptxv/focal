import argparse
import csv
import math
import os
import statistics

import torch

import focal

MAX_ABSOLUTE_ERROR_TOLERANCE = 5e-2
ROOT_MEAN_SQUARE_ERROR_TOLERANCE = 5e-3


CSV_FIELDS = [
    "backend",
    "device_name",
    "torch_version",
    "torch_cuda_version",
    "dtype",
    "batch_size",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "max_sequence_length",
    "warmup",
    "repeats",
    "p50_us",
    "p90_us",
    "mean_us",
    "min_us",
    "max_us",
    "max_absolute_error",
    "root_mean_square_error",
]


def parse_benchmark_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", choices=["cuda", "pytorch"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_heads", type=int, default=32)
    parser.add_argument("--key_value_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--csv")
    parser.add_argument("--plot_csv")
    parser.add_argument("--plot_png")
    return parser.parse_args()


def torch_dtype_from_cli_name(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def plot_requested(args):
    return args.plot_csv is not None or args.plot_png is not None


def check_benchmark_args(args):
    if plot_requested(args):
        if args.plot_csv is None or args.plot_png is None:
            raise SystemExit("--plot_csv and --plot_png must be passed together")
        return

    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.query_heads <= 0:
        raise SystemExit("--query_heads must be positive")
    if args.key_value_heads <= 0:
        raise SystemExit("--key_value_heads must be positive")
    if args.query_heads % args.key_value_heads != 0:
        raise SystemExit("--query_heads must be divisible by --key_value_heads")
    if args.head_dim <= 0:
        raise SystemExit("--head_dim must be positive")
    if args.max_sequence_length <= 0:
        raise SystemExit("--max_sequence_length must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.backend == "cuda" and args.head_dim != 128:
        raise SystemExit("backend=cuda requires --head_dim 128 for CUDA kernel v1")
    if args.backend == "cuda" and args.dtype != "fp16":
        raise SystemExit("backend=cuda requires --dtype fp16 for CUDA kernel v1")


def make_attention_inputs(args, dtype):
    query = torch.randn((args.batch_size, args.query_heads, args.head_dim), device="cuda", dtype=dtype)
    key_cache = torch.randn((args.batch_size, args.key_value_heads, args.max_sequence_length, args.head_dim), device="cuda", dtype=dtype)
    value_cache = torch.randn((args.batch_size, args.key_value_heads, args.max_sequence_length, args.head_dim), device="cuda", dtype=dtype)
    sequence_lengths = torch.full((args.batch_size,), args.max_sequence_length, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    return query, key_cache, value_cache, sequence_lengths, output


def make_cuda_runner(query, key_cache, value_cache, sequence_lengths, softmax_scale):
    def run():
        return focal.contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths, softmax_scale)

    return run


def make_pytorch_runner(args, query, key_cache, value_cache, output, softmax_scale):
    query_heads_per_key_value_head = args.query_heads // args.key_value_heads
    key_value_head_indices = torch.arange(args.query_heads, device="cuda", dtype=torch.int64) // query_heads_per_key_value_head

    query_float = query.float().contiguous().view(args.batch_size * args.query_heads, args.head_dim, 1)
    key_cache_float = key_cache.index_select(1, key_value_head_indices).float().contiguous().view(args.batch_size * args.query_heads, args.max_sequence_length, args.head_dim)
    value_cache_float = value_cache.index_select(1, key_value_head_indices).float().contiguous().view(args.batch_size * args.query_heads, args.max_sequence_length, args.head_dim)

    scores = torch.empty((args.batch_size * args.query_heads, args.max_sequence_length, 1), device="cuda", dtype=torch.float32)
    scores_2d = scores.view(args.batch_size * args.query_heads, args.max_sequence_length)
    max_score = torch.empty((args.batch_size * args.query_heads, 1), device="cuda", dtype=torch.float32)
    normalizer = torch.empty((args.batch_size * args.query_heads, 1), device="cuda", dtype=torch.float32)
    probabilities = torch.empty((args.batch_size * args.query_heads, args.max_sequence_length), device="cuda", dtype=torch.float32)
    probabilities_bmm = probabilities.view(args.batch_size * args.query_heads, 1, args.max_sequence_length)
    output_float = torch.empty((args.batch_size * args.query_heads, 1, args.head_dim), device="cuda", dtype=torch.float32)
    output_view = output_float.view(args.batch_size, args.query_heads, args.head_dim)

    def run():
        torch.bmm(key_cache_float, query_float, out=scores)
        scores_2d.mul_(softmax_scale)
        torch.amax(scores_2d, dim=1, keepdim=True, out=max_score)
        torch.sub(scores_2d, max_score, out=probabilities)
        torch.exp(probabilities, out=probabilities)
        torch.sum(probabilities, dim=1, keepdim=True, out=normalizer)
        probabilities.div_(normalizer)
        torch.bmm(probabilities_bmm, value_cache_float, out=output_float)
        output.copy_(output_view)

    return run


def attention_error_metrics(got, expected):
    difference = got.float() - expected.float()
    absolute_difference = difference.abs()
    root_mean_square_error = torch.sqrt(torch.mean(difference * difference)).item()
    return absolute_difference.max().item(), root_mean_square_error


def compare_runner_to_pytorch(args, run, query, key_cache, value_cache, sequence_lengths, output, softmax_scale):
    if args.max_sequence_length > 2048:
        raise SystemExit(
            "Correctness check is required before timing; this benchmark currently checks max_sequence_length <= 2048."
        )

    expected = focal.pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths, softmax_scale)
    if args.backend == "cuda":
        got = run()
    else:
        run()
        got = output
    torch.cuda.synchronize()
    return attention_error_metrics(got, expected)


def fail_on_attention_mismatch(max_absolute_error, root_mean_square_error):
    if (
        max_absolute_error > MAX_ABSOLUTE_ERROR_TOLERANCE
        or root_mean_square_error > ROOT_MEAN_SQUARE_ERROR_TOLERANCE
    ):
        raise SystemExit(
            "correctness check failed before timing: "
            f"max_absolute_error={max_absolute_error:.6g}, "
            f"root_mean_square_error={root_mean_square_error:.6g}"
        )


def time_attention_runner(run, warmup, repeats):
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


def summarize_latency(times):
    ordered = sorted(times)
    return {
        "p50_us": percentile(ordered, 0.50),
        "p90_us": percentile(ordered, 0.90),
        "mean_us": statistics.fmean(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def write_benchmark_csv(path, benchmark_row):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(benchmark_row)


def print_benchmark_table(benchmark_row):
    headers = ["backend", "dtype", "batch_size", "query_heads", "key_value_heads", "head_dim", "max_sequence_length", "p50_us", "p90_us", "mean_us"]
    display_cells = [
        benchmark_row["backend"],
        benchmark_row["dtype"],
        benchmark_row["batch_size"],
        benchmark_row["query_heads"],
        benchmark_row["key_value_heads"],
        benchmark_row["head_dim"],
        benchmark_row["max_sequence_length"],
        f"{benchmark_row['p50_us']:.3f}",
        f"{benchmark_row['p90_us']:.3f}",
        f"{benchmark_row['mean_us']:.3f}",
    ]
    widths = [max(len(str(header)), len(str(cell))) for header, cell in zip(headers, display_cells)]
    print("  ".join(str(h).rjust(w) for h, w in zip(headers, widths)))
    print("  ".join(str(cell).rjust(w) for cell, w in zip(display_cells, widths)))


def plot_benchmark_csv(csv_path, png_path):
    if not os.path.exists(csv_path):
        raise SystemExit(f"benchmark CSV does not exist: {csv_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required for plotting benchmark CSVs") from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting benchmark CSVs") from exc

    benchmark_rows = pd.read_csv(csv_path)
    if benchmark_rows.empty:
        raise SystemExit(f"benchmark CSV has no rows: {csv_path}")

    required_fields = {
        "backend",
        "batch_size",
        "query_heads",
        "key_value_heads",
        "head_dim",
        "max_sequence_length",
        "p50_us",
    }
    missing_fields = required_fields.difference(benchmark_rows.columns)
    if missing_fields:
        raise SystemExit(f"benchmark CSV missing required columns: {sorted(missing_fields)}")

    labels = [
        (
            f"{benchmark_row.backend} batch={benchmark_row.batch_size} "
            f"query_heads={benchmark_row.query_heads} kv_heads={benchmark_row.key_value_heads} "
            f"head_dim={benchmark_row.head_dim} max_seq={benchmark_row.max_sequence_length}"
        )
        for benchmark_row in benchmark_rows.itertuples(index=False)
    ]

    # Plot mode exists so CSV output and plotting stay in one benchmark entrypoint.
    figure, axis = plt.subplots()
    axis.bar(labels, benchmark_rows["p50_us"])
    axis.set_ylabel("p50 latency (us)")
    axis.set_xlabel("backend / shape")
    axis.set_title("contiguous_gqa_decode_attention benchmark")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    directory = os.path.dirname(png_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(png_path)


def main():
    args = parse_benchmark_args()
    check_benchmark_args(args)

    if plot_requested(args):
        plot_benchmark_csv(args.plot_csv, args.plot_png)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this benchmark on an NVIDIA GPU host.")
    if args.backend == "cuda" and not focal.cuda_extension_available():
        raise SystemExit(
            "focal CUDA extension is unavailable. Build it with: "
            "python -m pip install -e . --no-build-isolation"
        )

    dtype = torch_dtype_from_cli_name(args.dtype)
    torch.manual_seed(0)
    query, key_cache, value_cache, sequence_lengths, output = make_attention_inputs(args, dtype)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)

    if args.backend == "cuda":
        run = make_cuda_runner(query, key_cache, value_cache, sequence_lengths, softmax_scale)
    else:
        run = make_pytorch_runner(args, query, key_cache, value_cache, output, softmax_scale)

    max_absolute_error, root_mean_square_error = compare_runner_to_pytorch(
        args,
        run,
        query,
        key_cache,
        value_cache,
        sequence_lengths,
        output,
        softmax_scale,
    )
    fail_on_attention_mismatch(max_absolute_error, root_mean_square_error)
    times = time_attention_runner(run, args.warmup, args.repeats)
    stats = summarize_latency(times)

    benchmark_row = {
        "backend": args.backend,
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "key_value_heads": args.key_value_heads,
        "head_dim": args.head_dim,
        "max_sequence_length": args.max_sequence_length,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_absolute_error": max_absolute_error,
        "root_mean_square_error": root_mean_square_error,
        **stats,
    }

    print_benchmark_table(benchmark_row)
    if args.csv:
        write_benchmark_csv(args.csv, benchmark_row)


if __name__ == "__main__":
    main()
