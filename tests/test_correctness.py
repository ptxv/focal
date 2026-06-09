import sys

import pytest
import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03
MIN_M1_SPEEDUP = 1.5
MIN_M1_PREDEQUANT_SPEEDUP = 1.2


def max_errors(y, ref):
    diff = (y.float() - ref.float()).abs()
    max_abs_err = diff.max()
    max_rel_err = (diff / ref.float().abs().clamp_min(1.0)).max()
    return max_abs_err.item(), max_rel_err.item()


def time_cuda(fn, warmup, iters):
    # CUDA events catch real kernel regressions.
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


SHAPES = [
    (M, K, N)
    for M in [1, 2, 4, 8]
    for K in [4096, 8192]
    for N in [4096, 8192, 11008]
] + [(M, 11008, 4096) for M in [1, 2, 4, 8]]


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_w4a16_linear_correctness(M, K, N):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=1234)

    y = focal_w4a16.w4a16_linear(x, wq, scales, zeros)
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
    ref = (x.float() @ w_deq.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    max_abs_err, max_rel_err = max_errors(y, ref)
    sys.__stdout__.write(
        f"M={M} K={K} N={N} "
        f"max_abs_err={max_abs_err:.6f} max_rel_err={max_rel_err:.6f}\n"
    )
    sys.__stdout__.flush()

    assert max_abs_err <= MAX_ABS_ERR
    assert max_rel_err <= MAX_REL_ERR


def test_unhelpful_k11008_wide_n_is_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    x = torch.empty((1, 11008), device="cuda", dtype=torch.bfloat16)
    wq = torch.empty((11008, 11008 // 8), device="cuda", dtype=torch.int32)
    scales = torch.empty((11008, 11008 // 128), device="cuda", dtype=torch.float16)
    zeros = torch.empty((11008, 11008 // 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="unsupported K/N"):
        focal_w4a16.w4a16_linear(x, wq, scales, zeros)


def test_structured_pack_order_matches_direct_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    M, K, N = 2, 4096, 4096
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4321)
    x = 0.01 * torch.randn((M, K), device="cuda", dtype=torch.bfloat16, generator=gen)
    q = torch.arange(K, device="cuda", dtype=torch.int64)
    q = q.remainder(16).to(torch.uint8).repeat(N, 1).contiguous()
    wq = focal_w4a16.pack_int4_weight(q)
    scales = torch.full((N, K // 128), 0.01, device="cuda", dtype=torch.float16)
    zeros = torch.full((N, K // 128), 7.0, device="cuda", dtype=torch.float16)

    y = focal_w4a16.w4a16_linear(x.contiguous(), wq, scales, zeros)
    w_direct = (q.float() - zeros[0, 0].float()) * scales[0, 0].float()
    ref = (x.float() @ w_direct.T).to(torch.bfloat16)
    torch.cuda.synchronize()

    max_abs_err, max_rel_err = max_errors(y, ref)
    sys.__stdout__.write(
        f"structured_pack max_abs_err={max_abs_err:.6f} "
        f"max_rel_err={max_rel_err:.6f}\n"
    )
    sys.__stdout__.flush()

    assert max_abs_err <= MAX_ABS_ERR
    assert max_rel_err <= MAX_REL_ERR


def test_m1_kill_criterion_speed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    # This is the README kill criterion.
    M, K, N = 1, 4096, 4096
    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=1234)
    w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)

    focal_ms = time_cuda(
        lambda: focal_w4a16.w4a16_linear(x, wq, scales, zeros),
        warmup=20,
        iters=100,
    )
    baseline_ms = time_cuda(
        lambda: (x.float() @ focal_w4a16.dequant_ref(wq, scales, zeros, K).T).to(
            torch.bfloat16
        ),
        warmup=20,
        iters=100,
    )
    predequant_ms = time_cuda(
        lambda: (x.float() @ w_deq.T).to(torch.bfloat16),
        warmup=20,
        iters=100,
    )
    speedup = baseline_ms / focal_ms
    predequant_speedup = predequant_ms / focal_ms
    sys.__stdout__.write(
        f"M={M} K={K} N={N} focal_ms={focal_ms:.6f} "
        f"baseline_dequant_matmul_ms={baseline_ms:.6f} "
        f"predequant_matmul_ms={predequant_ms:.6f} "
        f"speedup={speedup:.3f} predequant_speedup={predequant_speedup:.3f}\n"
    )
    sys.__stdout__.flush()

    assert speedup >= MIN_M1_SPEEDUP
    assert predequant_speedup >= MIN_M1_PREDEQUANT_SPEEDUP
