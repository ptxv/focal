# Blog Notes

## Short Claim

This repo tests a narrow W4A16 decode GEMV kernel.

The best current claim is shape-specific.

On one H100, the kernel beats PyTorch dequant+matmul.

It also beats PyTorch predequant+matmul on all kept shapes.

The weakest kept predequant win is only 1.0069x.

That weak case is `M=8 K=8192 N=11008`.

Do not describe this as a general model speedup.

## Best Evidence

Hardware: NVIDIA H100 80GB HBM3.

CUDA test suite: 32 passed on the H100.

Shape sweep: 28 supported shapes.

Worst absolute error: 0.125.

Worst relative error: 0.007519.

Core shape: `M=1 K=4096 N=4096`.

Core latency: 0.00883680 ms.

Core speedup versus dequant+matmul: 51.12x.

Core speedup versus predequant+matmul: 4.27x.

Down-proj shape: `M=1 K=11008 N=4096`.

Down-proj latency: 0.01950592 ms.

Down-proj speedup versus dequant+matmul: 56.49x.

Down-proj speedup versus predequant+matmul: 3.86x.

Decode bundle latency: 0.13827360 ms.

Decode bundle speedup versus dequant+matmul: 37.16x.

Decode bundle speedup versus predequant+matmul: 2.82x.

Raw evidence:

- `docs/evidence/summary.json`
- `docs/evidence/results_kept.jsonl`
- `docs/evidence/decode_bundle_kept.json`
- `docs/evidence/ncu_m1_4096_4096_final.csv`
- `docs/evidence/ncu_m1_11008_4096_kept.csv`
- `docs/evidence/ncu_m8_8192_11008_final.csv`

## Useful NCU Facts

Core `M=1 K=4096 N=4096`:

- Duration: 15,840 ns.
- Registers per thread: 31.
- Memory throughput: 39.79 percent.
- DRAM throughput: 16.85 percent.
- SM throughput: 33.83 percent.
- Active warps: 21.96 percent.

Down-proj `M=1 K=11008 N=4096`:

- Duration: 35,552 ns.
- Registers per thread: 31.
- Memory throughput: 48.01 percent.
- DRAM throughput: 20.29 percent.
- SM throughput: 40.03 percent.
- Active warps: 23.15 percent.

Large `M=8 K=8192 N=11008`:

- Duration: 275,872 ns.
- Registers per thread: 56.
- Memory throughput: 83.21 percent.
- DRAM throughput: 5.45 percent.
- SM throughput: 41.81 percent.
- Active warps: 40.52 percent.

## Optimization Story

The kernel uses groupwise W4A16 dequantization.

That matches the broad AWQ/GPTQ serving contract.

It does not match Marlin or vLLM packed layouts.

The most important implementation choice is local ownership.

Each lane owns one int32 word containing eight int4 weights.

That avoids duplicate packed-weight loads inside one column group.

The dequant expression is algebraically rewritten.

`(q - zero) * scale` becomes `fmaf(q, scale, -zero_scaled)`.

That leaves one fused operation in the nibble loop.

The code keeps `M` as a compile-time template.

That keeps accumulators scalarized and visible to ptxas.

The kept kernels reported no spills.

The code uses warp shuffle reductions.

That avoids shared memory for the 16-lane column group.

`N_TILE=16` is intentionally modest.

It creates more CTAs for small decode projections.

The down-proj support is deliberately narrow.

Only `K=11008,N=4096` was added.

Wide `K=11008,N=11008` remains rejected.

## Failed Ideas

Symmetric-zero fast path was tested and removed.

It slowed `M=1 K=4096 N=4096`.

It also slowed `M=1 K=11008 N=4096`.

It only helped one larger row by 1.017x.

Evidence: `docs/evidence/results_symmetric.jsonl`.

Warp-shuffle experimentation was also rejected earlier.

Evidence: `docs/evidence/rejected_shuffle.jsonl`.

## Comparison Points

The current real comparisons are PyTorch baselines.

Baseline one: dequantize weights every call, then matmul.

Baseline two: predequantize once, then matmul.

The predequant baseline is stricter for kernel latency.

It removes per-call dequantization cost.

Marlin is the closest external research comparison.

Marlin targets FP16xINT4 LLM inference.

It reports near-ideal 4x speedups at batch sizes 16-32.

Source: https://github.com/IST-DASLab/marlin

Paper: https://arxiv.org/abs/2408.11743

Marlin is not a fair numeric comparison yet.

It uses its own offline packing and layout transforms.

Its README also says Hopper is not optimized yet.

vLLM AWQ-Marlin is a relevant serving comparison.

It has explicit pack factors, group sizes, and zero points.

Source: https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/quantization/awq_marlin.html

TensorRT-LLM is the production W4A16 comparison target.

It documents W4A16 AWQ and W4A16 GPTQ support.

Source: https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/quantization.html

bitsandbytes Linear4bit is not contract-equivalent.

Its public API centers NF4 and FP4 module storage.

Source: https://huggingface.co/docs/bitsandbytes/v0.43.3/en/reference/nn/linear4bit

## Fair Future Comparison

First integrate one real checkpoint packing format.

AWQ with group size 128 is the natural first target.

Then compare one layer through identical tensors.

Use the same activation dtype, weight values, and scales.

Compare against vLLM AWQ-Marlin if it accepts the model.

Compare against TensorRT-LLM if engine build succeeds.

Compare against Marlin only after matching its pack format.

Report both isolated layer latency and decode-bundle latency.

Do not compare against full-model tokens per second first.

Full-model serving adds scheduler and attention effects.
