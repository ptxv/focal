import sys

import pytest
import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03


def max_errors(y, ref):
    diff = (y.float() - ref.float()).abs()
    max_abs_err = diff.max()
    max_rel_err = (diff / ref.float().abs().clamp_min(1.0)).max()
    return max_abs_err.item(), max_rel_err.item()


@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("K", [4096, 8192])
@pytest.mark.parametrize("N", [4096, 8192])
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
