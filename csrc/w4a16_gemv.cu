#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>


namespace {

constexpr int N_TILE = 16;
constexpr int THREADS = 256;
constexpr int COL_THREADS = THREADS / N_TILE;

// This kernel targets groupwise W4A16 decode GEMV.
// TensorRT-LLM exposes W4A16 AWQ and GPTQ paths.
// https://nvidia.github.io/TensorRT-LLM/1.2.0rc4/features/quantization.html
// vLLM AWQ-Marlin tracks bits, group_size, and zero_point.
// https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/quantization/awq_marlin.html
// We implement the same broad contract, not their packing.
// This repo keeps weights as row-major int32 packed nibbles.
// That makes PyTorch dequant baselines fairer than Marlin timings.
//
// Marlin describes a "FP16xINT4 matmul kernel aimed at LLM inference".
// https://github.com/IST-DASLab/marlin
// It also stresses static offsets and PTX review.
// We borrowed that narrow idea, not Marlin's tensor-core layout.
// Our small-M GEMV uses scalar lanes and warp reductions instead.
// The gap is intentional until a real Marlin packer lands.

template <int M>
__global__ void w4a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ wq,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __nv_bfloat16* __restrict__ y,
    int K,
    int N) {
  const int tid = threadIdx.x;
  const int col_local = tid / COL_THREADS;
  const int lane = tid & (COL_THREADS - 1);
  const int n = blockIdx.x * N_TILE + col_local;
  const int words_per_row = K >> 3;
  const int groups_per_row = K >> 7;

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] = 0.0f;
  }

  const int w_base = n * words_per_row;
  const int group_base = n * groups_per_row;

  // H100 has many SMs, so N_TILE stays modest.
  // More CTAs keep small-N decode from underfilling SMs.
  // The H100 guide favors occupancy with low register pressure.
  // See docs/evidence/ncu_m1_11008_4096_kept.csv.
  // N_TILE=16 kept 31 registers for M=1.
  //
  // Marlin uses offline reshuffling for ideal access patterns.
  // This repo does not yet own a real checkpoint packer.
  // Instead, each lane owns one existing packed word.
  // That local adjustment removes duplicate row-major weight loads.
  // It also keeps nibble order directly testable.
  for (int group = 0; group < groups_per_row; ++group) {
    const float scale = __half2float(scales[group_base + group]);
    const float zero = __half2float(zeros[group_base + group]);
    // Dequant math follows w=(q-zero)*scale.
    // We rewrite it as fmaf(q, scale, -zero_scaled).
    // This leaves one fused op inside the nibble loop.
    // The symmetric-zero variant removed zero loads.
    // It lost on core and down-proj H100 timings.
    // See docs/evidence/results_symmetric.jsonl.
    const float zero_scaled = zero * scale;
    const int k_base = (group << 7) + (lane << 3);
    // Marlin recommends static offsets and unrolled loops.
    // We interpret that as fixed group and lane strides.
    // The implementation differs from Marlin's striped tiling.
    // It only removes integer arithmetic in this simple layout.
    const uint32_t word =
        static_cast<uint32_t>(wq[w_base + (group << 4) + lane]);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q = static_cast<int>((word >> (4 * i)) & 0xF);
      const float w = fmaf(static_cast<float>(q), scale, -zero_scaled);

#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m] = fmaf(__bfloat162float(x[m * K + k_base + i]), w, acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
    // Warp shuffle reduction avoids shared-memory traffic.
    // One 16-lane column group matches one scale group.
    // This is the H100 guide's warp-reduction pattern.
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 8, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 4, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 2, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 1, COL_THREADS);
  }

  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      y[m * N + n] = __float2bfloat16(acc[m]);
    }
  }
}


void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}


void check_shape(
    int64_t M64,
    int64_t K64,
    int64_t N64,
    const torch::Tensor& wq,
    const torch::Tensor& scales) {
  TORCH_CHECK(M64 == 1 || M64 == 2 || M64 == 4 || M64 == 8, "unsupported M");
  TORCH_CHECK(N64 == 4096 || N64 == 8192 || N64 == 11008, "unsupported N");
  // Down-proj support is deliberately narrow.
  // K=11008,N=4096 matched Llama-2-7B down projection.
  // K=11008,N=11008 had weak predequant evidence.
  // The guard prevents unsupported shapes looking validated.
  TORCH_CHECK(
      K64 == 4096 || K64 == 8192 || (K64 == 11008 && N64 == 4096),
      "unsupported K/N");
  TORCH_CHECK(K64 % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(N64 % 64 == 0, "N must be divisible by 64");
  TORCH_CHECK(wq.size(1) == K64 / 8, "wq must have shape [N, K / 8]");
  TORCH_CHECK(scales.size(0) == N64 && scales.size(1) == K64 / 128,
              "scales must have shape [N, K / 128]");
}


torch::Tensor launch_w4a16(
    torch::Tensor x,
    torch::Tensor wq,
    torch::Tensor scales,
    torch::Tensor zeros) {
  check_tensor(x, "x", at::kBFloat16);
  check_tensor(wq, "wq", at::kInt);
  check_tensor(scales, "scales", at::kHalf);
  check_tensor(zeros, "zeros", at::kHalf);

  TORCH_CHECK(x.dim() == 2, "x must have shape [M, K]");
  TORCH_CHECK(wq.dim() == 2, "wq must have shape [N, K / 8]");
  TORCH_CHECK(scales.dim() == 2, "scales must have shape [N, K / 128]");
  TORCH_CHECK(zeros.dim() == 2, "zeros must have shape [N, K / 128]");

  TORCH_CHECK(wq.device() == x.device(), "wq must be on the same device as x");
  TORCH_CHECK(scales.device() == x.device(), "scales must be on the same device as x");
  TORCH_CHECK(zeros.device() == x.device(), "zeros must be on the same device as x");

  const int64_t M64 = x.size(0);
  const int64_t K64 = x.size(1);
  const int64_t N64 = wq.size(0);

  check_shape(M64, K64, N64, wq, scales);
  TORCH_CHECK(zeros.size(0) == N64 && zeros.size(1) == K64 / 128,
              "zeros must have shape [N, K / 128]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto y = torch::empty({M64, N64}, x.options());

  const int M = static_cast<int>(M64);
  const int K = static_cast<int>(K64);
  const int N = static_cast<int>(N64);

  const auto* x_ptr =
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wq_ptr = wq.data_ptr<int32_t>();
  const auto* scales_ptr =
      reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const auto* zeros_ptr =
      reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>());
  auto* y_ptr = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

  const dim3 grid(N / N_TILE);
  const dim3 block(THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Compile-time M keeps accumulators scalarized.
  // It prevents dynamic loops from hiding register pressure.
  // ptxas reported no spills for all kept variants.
  switch (M) {
    case 1:
      w4a16_gemv_kernel<1><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 2:
      w4a16_gemv_kernel<2><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 4:
      w4a16_gemv_kernel<4><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 8:
      w4a16_gemv_kernel<8><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


}  // namespace


torch::Tensor w4a16_linear_cuda(
    torch::Tensor x,
    torch::Tensor wq,
    torch::Tensor scales,
    torch::Tensor zeros) {
  return launch_w4a16(x, wq, scales, zeros);
}
