#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>


namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 128;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    // Four warp sums are enough because this v1 kernel fixes 128 threads.
    val = threadIdx.x < 4 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void decode_attn_contig_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int32_t* __restrict__ seq_lens,
    __half* __restrict__ out,
    int Hq,
    int Hkv,
    int L,
    float sm_scale) {
    const int d = threadIdx.x;
    const int block = blockIdx.x;
    const int b = block / Hq;
    const int hq = block - b * Hq;
    const int group_size = Hq / Hkv;
    const int hkv = hq / group_size;
    const int seq_len = seq_lens[b];

    const long long q_base = (static_cast<long long>(b) * Hq + hq) * kHeadDim;
    const long long kv_base = (static_cast<long long>(b) * Hkv + hkv) * L * kHeadDim;
    const long long out_base = q_base;

    const bool bad_seq_len = seq_len < 0 || seq_len > L;
    if (bad_seq_len || seq_len == 0) {
        // Guard invalid lengths in-device so the hot path does not sync to validate.
        out[out_base + d] = __float2half(bad_seq_len ? CUDART_NAN_F : 0.0f);
        return;
    }

    __shared__ float m_shared;
    __shared__ float l_shared;
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    // One block owns one output vector, so thread 0 owns the online-softmax state.
    if (d == 0) {
        m_shared = -INFINITY;
        l_shared = 0.0f;
    }
    __syncthreads();

    const float q_d = __half2float(q[q_base + d]);
    float acc = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        const long long kv_idx = kv_base + (static_cast<long long>(t) * kHeadDim + d);
        const float score_sum = block_reduce_sum(q_d * __half2float(k[kv_idx]));

        if (d == 0) {
            const float score = score_sum * sm_scale;
            if (t == 0) {
                alpha_shared = 0.0f;
                beta_shared = 1.0f;
                l_shared = 1.0f;
                m_shared = score;
            } else {
                const float m_new = fmaxf(m_shared, score);
                // Rescale the running sum when the max changes to keep softmax stable.
                alpha_shared = __expf(m_shared - m_new);
                beta_shared = __expf(score - m_new);
                l_shared = l_shared * alpha_shared + beta_shared;
                m_shared = m_new;
            }
        }
        __syncthreads();

        acc = acc * alpha_shared + beta_shared * __half2float(v[kv_idx]);
    }

    out[out_base + d] = __float2half(acc / l_shared);
}

void check_common(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& seq_lens) {
    CHECK_CUDA(q);
    CHECK_CUDA(k);
    CHECK_CUDA(v);
    CHECK_CUDA(seq_lens);
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(seq_lens);
    CHECK_HALF(q);
    CHECK_HALF(k);
    CHECK_HALF(v);
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");

    TORCH_CHECK(q.dim() == 3, "q must have rank 3 [B, Hq, D]");
    TORCH_CHECK(k.dim() == 4, "k must have rank 4 [B, Hkv, L, D]");
    TORCH_CHECK(v.dim() == 4, "v must have rank 4 [B, Hkv, L, D]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have rank 1 [B]");

    const auto B = q.size(0);
    const auto Hq = q.size(1);
    const auto D = q.size(2);
    const auto Hkv = k.size(1);
    const auto L = k.size(2);

    TORCH_CHECK(Hq > 0, "Hq must be positive");
    TORCH_CHECK(Hkv > 0, "Hkv must be positive");
    TORCH_CHECK(D == kHeadDim, "D must equal 128 for CUDA kernel v1");
    TORCH_CHECK(k.device() == q.device(), "k must be on the same CUDA device as q");
    TORCH_CHECK(v.device() == q.device(), "v must be on the same CUDA device as q");
    TORCH_CHECK(seq_lens.device() == q.device(), "seq_lens must be on the same CUDA device as q");
    TORCH_CHECK(k.size(0) == B, "k batch dimension must match q");
    TORCH_CHECK(v.size(0) == B, "v batch dimension must match q");
    TORCH_CHECK(v.size(1) == Hkv, "k and v Hkv dimensions must match");
    TORCH_CHECK(v.size(2) == L, "k and v sequence dimensions must match");
    TORCH_CHECK(k.size(3) == D, "k head dimension must match q");
    TORCH_CHECK(v.size(3) == D, "v head dimension must match q");
    TORCH_CHECK(seq_lens.size(0) == B, "seq_lens must have shape [B]");
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
}

void check_int_size(const char* name, int64_t value) {
    TORCH_CHECK(value >= 0, name, " must be non-negative");
    TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " must fit in int32 for CUDA kernel v1");
}

void check_out(const torch::Tensor& out, const torch::Tensor& q) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    CHECK_HALF(out);
    TORCH_CHECK(out.dim() == 3, "out must have rank 3 [B, Hq, D]");
    TORCH_CHECK(out.device() == q.device(), "out must be on the same CUDA device as q");
    TORCH_CHECK(out.size(0) == q.size(0), "out batch dimension must match q");
    TORCH_CHECK(out.size(1) == q.size(1), "out Hq dimension must match q");
    TORCH_CHECK(out.size(2) == q.size(2), "out head dimension must match q");
}

void check_no_output_alias(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& seq_lens,
    const torch::Tensor& out) {
    TORCH_CHECK(!out.is_alias_of(q), "out must not alias q");
    TORCH_CHECK(!out.is_alias_of(k), "out must not alias k");
    TORCH_CHECK(!out.is_alias_of(v), "out must not alias v");
    TORCH_CHECK(!out.is_alias_of(seq_lens), "out must not alias seq_lens");
}

void launch_decode_attn_contig(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& seq_lens,
    torch::Tensor& out,
    double sm_scale) {
    check_common(q, k, v, seq_lens);
    check_out(out, q);
    check_no_output_alias(q, k, v, seq_lens, out);
    TORCH_CHECK(std::isfinite(sm_scale), "sm_scale must be finite");

    c10::cuda::CUDAGuard device_guard(q.device());

    check_int_size("B", q.size(0));
    check_int_size("Hq", q.size(1));
    check_int_size("Hkv", k.size(1));
    check_int_size("L", k.size(2));
    TORCH_CHECK(
        q.size(0) * q.size(1) <= std::numeric_limits<int>::max(),
        "B * Hq must fit in int32 grid dimension for CUDA kernel v1");

    const int B = static_cast<int>(q.size(0));
    const int Hq = static_cast<int>(q.size(1));
    const int Hkv = static_cast<int>(k.size(1));
    const int L = static_cast<int>(k.size(2));

    if (B == 0) {
        return;
    }

    const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const __half* v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    const int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
    __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    const dim3 grid(B * Hq);
    const dim3 block(kThreads);
    decode_attn_contig_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        seq_lens_ptr,
        out_ptr,
        Hq,
        Hkv,
        L,
        static_cast<float>(sm_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


torch::Tensor decode_attn_contig_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seq_lens,
    double sm_scale) {
    auto out = torch::empty_like(q);
    launch_decode_attn_contig(q, k, v, seq_lens, out, sm_scale);
    return out;
}


torch::Tensor decode_attn_contig_out_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seq_lens,
    torch::Tensor out,
    double sm_scale) {
    launch_decode_attn_contig(q, k, v, seq_lens, out, sm_scale);
    return out;
}
