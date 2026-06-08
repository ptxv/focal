import sys

import pytest
import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03
MIN_M1_SPEEDUP = 1.5


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


@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("N", [4096, 8192, 11008])
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


def test_m1_kill_criterion_speed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    M, K, N = 1, 4096, 4096
    x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=1234)

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
    speedup = baseline_ms / focal_ms
    sys.__stdout__.write(
        f"M={M} K={K} N={N} focal_ms={focal_ms:.6f} "
        f"baseline_dequant_matmul_ms={baseline_ms:.6f} speedup={speedup:.3f}\n"
    )
    sys.__stdout__.flush()

    assert speedup >= MIN_M1_SPEEDUP
