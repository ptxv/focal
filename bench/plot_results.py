import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("png_path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv_path):
        raise SystemExit(f"benchmark CSV does not exist: {args.csv_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required for plotting benchmark CSVs") from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting benchmark CSVs") from exc

    df = pd.read_csv(args.csv_path)
    if df.empty:
        raise SystemExit(f"benchmark CSV has no rows: {args.csv_path}")

    required = {"backend", "B", "Hq", "Hkv", "D", "L", "p50_us"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"benchmark CSV missing required columns: {sorted(missing)}")

    labels = [
        f"{row.backend} B{row.B} Hq{row.Hq} Hkv{row.Hkv} D{row.D} L{row.L}"
        for row in df.itertuples(index=False)
    ]

    fig, ax = plt.subplots()
    ax.bar(labels, df["p50_us"])
    ax.set_ylabel("p50 latency (us)")
    ax.set_xlabel("backend / shape")
    ax.set_title("decode_attn_contig benchmark")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    directory = os.path.dirname(args.png_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(args.png_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
