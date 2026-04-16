#if defined(ENABLE_NVIDIA_API)
#include "kernel.cuh"
#include "../core/utils.h"

template <typename scalar_t, typename Tdata>
void awq_marlin_gemm_kernel(
    const void *a,
    void *c,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *a_scales,
    void *global_scale,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int size_m,
    int size_k,
    int size_n,
    int b_q_size_0,
    int b_q_size_1,
    int a_stride_0,
    int b_zeros_size_1,
    int num_groups,
    cudaStream_t stream)
{
    // scalar_t *a, Tdata *b_scales
    vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

    if constexpr (std::is_same<scalar_t, half>::value)
    {
        a_type_id = vllm::kFloat16.id();
        c_type_id = vllm::kFloat16.id();
    }
    else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value)
    {
        a_type_id = vllm::kBFloat16.id();
        c_type_id = vllm::kBFloat16.id();
    }
    else
    {
        // 此时c和b_scales类型相同
        if constexpr (std::is_same<Tdata, half>::value)
        {
            c_type_id = vllm::kFloat16.id();
        }
        else if constexpr (std::is_same<Tdata, nv_bfloat16>::value)
        {
            c_type_id = vllm::kBFloat16.id();
        }
        else
        {
            c_type_id = vllm::kBFloat16.id();
            RUNTIME_CHECK(c != nullptr, "c must be passed for W4A8-FP4\n");
        }
        if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value)
        {
            a_type_id = vllm::kFE4M3fn.id();
        }
        else if constexpr (std::is_same<scalar_t, char>::value)
        {
            a_type_id = vllm::kS8.id();
        }
        else
        {
            RUNTIME_CHECK(false, "unsupported `a` scalar_type\n");
        }
    }

    s_type_id = c_type_id;
    if (b_type_id == vllm::kFE2M1f.id())
    {
        if constexpr (std::is_same<Tdata, __nv_fp8_e4m3>::value)
        {
            s_type_id = vllm::kFE4M3fn.id();
        }
        else if constexpr (std::is_same<Tdata, uint8_t>::value)
        {
            printf("b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu\n");
            s_type_id = vllm::kFE8M0fnu.id();
        }
        else
        {
            RUNTIME_CHECK(false,
                          "When b_type = float4_e2m1f, b_scale scalar type must be",
                          "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
        }
    }

    vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
    vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
    vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
    vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

    int pack_factor = 32 / b_type.size_bits();

    // Verify a = [size_m, size_k]

    // Verify b
    RUNTIME_CHECK(
        size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    RUNTIME_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_size_0,
                  "Shape mismatch: b_q_weight.size(0) = ", b_q_size_0,
                  ", size_k = ", size_k,
                  ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    RUNTIME_CHECK(
        b_q_size_1 % MARLIN_NAMESPACE_NAME::tile_size == 0,
        "b_q_weight.size(1) = ", b_q_size_1,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    int actual_size_n =
        (b_q_size_1 / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
    RUNTIME_CHECK(size_n == actual_size_n, "size_n = ", size_n,
                  ", actual_size_n = ", actual_size_n);

    // Verify device and strides

    // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
    RUNTIME_CHECK(a_stride_0 % 8 == 0, "A.stride(0) must divisible by 8");
    RUNTIME_CHECK(reinterpret_cast<uintptr_t>(a) % 16 == 0, "A must aligned to 16 bytes");

    if (a_scales != nullptr)
    {

        RUNTIME_CHECK(a_type.size_bits() == 8,
                      "a_scales can only be used for 8bit activation.");
    }
    else
    {
        RUNTIME_CHECK(a_type.size_bits() != 8,
                      "the a_scales parameter must be passed for 8bit activation.");
    }

    int device_id = 0;
    // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
    // auto -1)
    int thread_k = -1;
    // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
    // auto -1)
    int thread_n = -1;
    // sms: number of SMs to use for the kernel
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

    // Alloc buffers
    float *c_tmp = nullptr;
    void *a_tmp = nullptr;
    void *workspace = nullptr;
    void *total_buffer = nullptr;
    int c_tmp_bytes = 0;
    // Alloc C tmp buffer that is going to be used for the global reduce

    if (use_fp32_reduce)
    {
        int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
        max_m_block_size = min(max_m_block_size, 64);
        int max_c_tmp_size =
            sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
        c_tmp_bytes = max_c_tmp_size * sizeof(float);
    }

    // Detect groupsize and act_order

    // b_scales = [num_groups, size_n]
    // g_idx.size(-1) == size_k && perm.size(-1) == size_k
    int a_tmp_bytes = 0;
    bool has_act_order = false;

    if (g_idx != nullptr && perm != nullptr)
    {
        has_act_order = true;
    }
    int group_size = -1;
    if (has_act_order)
    {
        a_tmp_bytes = size_m * size_k * sizeof(scalar_t);
        if (is_k_full)
        {
            RUNTIME_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
            RUNTIME_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                          ", is not divisible by num_groups = ", num_groups);
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = 0;
        }
    }
    else
    {
        if (num_groups > 1)
        {
            RUNTIME_CHECK(
                size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(0) = ", num_groups);
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = -1;
        }
    }

    int workspace_bytes = sms * sizeof(int64_t);
    const int total_bytes = c_tmp_bytes + a_tmp_bytes + workspace_bytes;
    // ===================== 3. 单次 cudaMalloc 分配 =====================
    if (total_bytes > 0)
    {
        cudaMalloc(&total_buffer, total_bytes);
        // 把 workspace 初始化为 0（唯一需要 memset 的部分）
        cudaMemset(total_buffer, 0, total_bytes);
    }
    // ===================== 4. 手动切分指针（核心！） =====================
    uint8_t *ptr = reinterpret_cast<uint8_t *>(total_buffer);
    // 分配 c_tmp
    if (use_fp32_reduce && c_tmp_bytes > 0)
    {
        c_tmp = reinterpret_cast<float *>(ptr);
        ptr += c_tmp_bytes;
    }
    // 分配 a_tmp
    if (has_act_order && a_tmp_bytes > 0)
    {
        a_tmp = ptr;
        ptr += a_tmp_bytes;
    }

    // 分配 workspace
    if (workspace_bytes > 0)
    {
        workspace = ptr;
        ptr += workspace_bytes;
    }

    if (global_scale != nullptr)
    {

        RUNTIME_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                      "global_scale can only be used for nvfp4 format.");
    }
    else
    {
        RUNTIME_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                      "the global_scale parameter must be passed for nvfp4 format.");
    }
    // b_bias = [size_n, 1]
    bool has_bias = (b_bias != nullptr);

    bool has_zp = (b_zeros != nullptr);
    if (has_zp)
    {
        RUNTIME_CHECK(
            b_type == vllm::kU4 || b_type == vllm::kU8,
            "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());
    }
    else
    {
        RUNTIME_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                          b_type == vllm::kS4 || b_type == vllm::kS8 ||
                          b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                      "b_type must be uint4b8, uint8b128, int4, int8, "
                      "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                      b_type.str());
    }

    if (has_zp && is_zp_float)
    {
        if constexpr (!std::is_same<scalar_t, half>::value)
        {
            printf("Computation a_type must be float16 (half) when using float zero "
                   "points.\n");
        }
    }

    // Verify b_zeros
    if (has_zp)
    {
        if (is_zp_float)
        {
            // b_zeros = [num_groups, size_n]
            RUNTIME_CHECK(b_zeros_size_1 == size_n,
                          "b_zeros dim 1 = ", b_zeros_size_1,
                          " is not size_n = ", size_n);
            RUNTIME_CHECK(num_groups != -1, "num_groups must be != -1");
        }
        else
        {

            RUNTIME_CHECK(b_zeros_size_1 == size_n / pack_factor,
                          "b_zeros dim 1 = ", b_zeros_size_1,
                          " is not size_n / pack_factor = ", size_n / pack_factor);
        }
    }

    // Verify workspace size
    RUNTIME_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                  "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                  MARLIN_NAMESPACE_NAME::min_thread_n);

    // a_scales和global_scale都必须是float *

    if (a_type.size_bits() == 16)
    {
        RUNTIME_CHECK((a_type == c_type), "scalar type of a must be the same with c for 16 bit activation\n");
    }

    marlin::marlin_mm(
        a, b_q_weight, c, c_tmp,
        b_bias, a_scales, b_scales,
        global_scale, b_zeros, g_idx,
        perm, a_tmp, size_m, size_n, size_k, a_stride_0,
        workspace, a_type, b_type, c_type, s_type, has_bias,
        has_act_order, is_k_full, has_zp, num_groups, group_size, device_id,
        stream, thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
    cudaFree(total_buffer);
}

enum DataType
{
    DT_FLOAT16 = 0,
    DT_BFLOAT16 = 1,
};

extern "C" void awq_marlin_gemm_nv(void *c,
                                   const void *a,
                                   const void *b_q_weight,
                                   void *b_bias,
                                   void *b_scales,
                                   void *a_scales,
                                   void *global_scale,
                                   void *b_zeros,
                                   void *g_idx,
                                   void *perm,
                                   int64_t b_type_id,
                                   bool is_k_full,
                                   bool use_atomic_add,
                                   bool use_fp32_reduce,
                                   bool is_zp_float,
                                   int size_m,
                                   int size_k,
                                   int size_n,
                                   int b_q_size_0,
                                   int b_q_size_1,
                                   int a_stride_0,
                                   int b_zeros_size_1,
                                   int num_groups,
                                   int dataType)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }
    switch (static_cast<DataType>(dataType))
    {
    case DT_FLOAT16:
        awq_marlin_gemm_kernel<__half, __half>(
            a, c,
            b_q_weight,
            b_bias, b_scales,
            a_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            b_type_id,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            static_cast<int64_t>(size_m),
            static_cast<int64_t>(size_k),
            static_cast<int64_t>(size_n),
            static_cast<int64_t>(b_q_size_0),
            static_cast<int64_t>(b_q_size_1),
            static_cast<int64_t>(a_stride_0),
            static_cast<int64_t>(b_zeros_size_1),
            num_groups,
            stream);
        break;
    case DT_BFLOAT16:
        awq_marlin_gemm_kernel<__nv_bfloat16, __nv_bfloat16>(
            a, c,
            b_q_weight,
            b_bias, b_scales,
            a_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            b_type_id,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            static_cast<int64_t>(size_m),
            static_cast<int64_t>(size_k),
            static_cast<int64_t>(size_n),
            static_cast<int64_t>(b_q_size_0),
            static_cast<int64_t>(b_q_size_1),
            static_cast<int64_t>(a_stride_0),
            static_cast<int64_t>(b_zeros_size_1),
            num_groups,
            stream);
        break;
    default:
        printf("Unsupported datatype\n");
        break;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}
#endif
