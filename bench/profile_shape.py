import argparse

import torch

import focal_w4a16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--profile-target-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    x, wq, scales, zeros = focal_w4a16.random_case(args.M, args.K, args.N, 1234)
    if args.profile_target_only:
        for _ in range(args.calls):
            focal_w4a16.w4a16_linear(x, wq, scales, zeros)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        focal_w4a16.w4a16_linear(x, wq, scales, zeros)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    for _ in range(args.calls):
        focal_w4a16.w4a16_linear(x, wq, scales, zeros)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
