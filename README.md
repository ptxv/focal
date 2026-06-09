# focal-w4a16

Brutally small CUDA benchmark repo for one operator: small-M W4A16 linear with bf16 activations, int4 packed weights, fp16 group scales/zeros, fp32 accumulation, and bf16 output.

## Hardware Target

- 1x NVIDIA A100 40GB, compute capability 8.0
- 1x NVIDIA H100 80GB, compute capability 9.0

## Build

```bash
export TORCH_CUDA_ARCH_LIST="8.0 9.0"
pip install -v -e .
```

## Test

```bash
python -m pytest -q
```

## Benchmark

```bash
python bench/bench_w4a16.py
```

Optional benchmark knobs:

```bash
python bench/bench_w4a16.py --warmup 50 --iters 200 --jsonl bench/results.jsonl
```

## H100 Report

Open `docs/report.html` for the current H100 evidence summary.

The raw benchmark, NCU, SASS, and research files live in `docs/evidence/`.

## Shape Contract

The Python API is:

```python
focal_w4a16.w4a16_linear(x, wq, scales, zeros) -> y
```

Inputs are CUDA, contiguous, row-major tensors:

- `x`: `[M, K]`, `torch.bfloat16`
- `wq`: `[N, K / 8]`, `torch.int32`
- `scales`: `[N, K / 128]`, `torch.float16`
- `zeros`: `[N, K / 128]`, `torch.float16`
- `y`: `[M, N]`, `torch.bfloat16`

Supported MVP shapes:

- `M in {1, 2, 4, 8}`
- `K in {4096, 8192}`
- `N in {4096, 8192, 11008}`
- Extra down-proj shape: `K = 11008, N = 4096`
- `K % 128 == 0`
- `N % 64 == 0`

Weight packing uses one `int32` per eight unsigned int4 values:

```text
word = Wq[n, k / 8]
q = (word >> (4 * (k % 8))) & 0xF
group = k / 128
w = (float(q) - float(zeros[n, group])) * float(scales[n, group])
```

## Continue/Kill Criteria

Continue if:
  - For M <= 4, focal is <= 0.55x latency of baseline_dequant_matmul on at least one of A100 or H100.
  - For M = 8, focal is <= 0.70x latency of baseline_dequant_matmul on at least one of A100 or H100.
  - Correctness passes all tested shapes.

Kill if:
  - M=1,K=4096,N=4096 is not at least 1.5x faster than baseline_dequant_matmul.
  - Kernel is slower than baseline_dequant_matmul for most M<=4 cases.
  - Correctness requires shape-specific hacks.
  - The implementation grows into a framework.

## SASS Inspection

```bash
bash scripts/inspect_sass.sh
```

The script finds the built extension, runs `cuobjdump --dump-sass` when available, stores full SASS, and stores a grep summary for `LDG`, `STG`, `BF16`, `FMA`, and `IMAD`.

## Known Limitations

- CUDA only; there is no CPU fallback.
- The MVP kernel supports only the listed shapes and contiguous layouts.
- The CUDA kernel does not use Triton, CUTLASS, shared-memory tiling, or autotuning.
- Benchmark claims are valid only for the exact GPU, shape, software stack, and command output shown by the benchmark.
