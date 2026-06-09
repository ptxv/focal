import pytest
import torch

import focal_w4a16


MAX_ABS_ERR = 0.25
MAX_REL_ERR = 0.03
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


def test_llama2_7b_supported_decode_bundle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    for index, (name, M, K, N) in enumerate(LAYERS):
        x, wq, scales, zeros = focal_w4a16.random_case(M, K, N, seed=5000 + index)
        y = focal_w4a16.w4a16_linear(x, wq, scales, zeros)
        w_deq = focal_w4a16.dequant_ref(wq, scales, zeros, K)
        ref = (x.float() @ w_deq.T).to(torch.bfloat16)
        torch.cuda.synchronize()
        max_abs_err, max_rel_err = max_errors(y, ref)

        assert max_abs_err <= MAX_ABS_ERR, name
        assert max_rel_err <= MAX_REL_ERR, name
