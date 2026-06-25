#include "kernel.cuh"

template <typename scalar_t, typename Tdata>
void vllm_moe_wna16_marlin_gemm(
    const void *a, void *c,
    const void *b_q_weight,
    void *b_bias, void *b_scales,
    void *a_scales,     // float *a_scales
    void *global_scale, // float *global_scale
    void *b_zeros,
    void *g_idx,
    void *perm,
    void *sorted_token_ids, void *expert_ids,
    void *num_tokens_past_padded, void *topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm,
    int64_t sorted_token_ids_size_0,
    int64_t b_q_weight_size_0,
    int64_t b_q_weight_size_1,
    int64_t b_q_weight_size_2,
    int64_t b_scales_size_1,
    int64_t b_scales_size_2,
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
    else if (b_type_id == vllm::kFE4M3fn.id() &&
             std::is_same<Tdata, uint8_t>::value)
    {
        s_type_id = vllm::kFE8M0fnu.id();
    }

    vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
    vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
    vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
    vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

    int pack_factor = 32 / b_type.size_bits();
    int num_experts = b_q_weight_size_0;

    if (moe_block_size != 8)
    {
        RUNTIME_CHECK(moe_block_size % 16 == 0,
                      "unsupported moe_block_size=", moe_block_size);
        RUNTIME_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                      "unsupported moe_block_size=", moe_block_size);
    }

    // Verify a = [size_m, size_k]

    // Verify B
    RUNTIME_CHECK(
        size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    RUNTIME_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight_size_1,
                  "Shape mismatch: b_q_weight.size(1) = ", b_q_weight_size_1,
                  ", size_k = ", size_k,
                  ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    RUNTIME_CHECK(
        b_q_weight_size_2 % MARLIN_NAMESPACE_NAME::tile_size == 0,
        "b_q_weight.size(2) = ", b_q_weight_size_2,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    int actual_size_n =
        (b_q_weight_size_2 / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
    RUNTIME_CHECK(size_n == actual_size_n, "size_n = ", size_n,
                  ", actual_size_n = ", actual_size_n);

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
    // sms: number of SMs to use for the kernel
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

    // c_size = [size_m * top_k, size_n]
    //  Alloc buffers

    // Alloc C tmp buffer that is going to be used for the global reduce

    int c_tmp_bytes = 0;
    if (use_fp32_reduce && !use_atomic_add)
    {
        // max num of threadblocks is sms * 4
        long max_c_tmp_size = min(
            (long)size_n * sorted_token_ids_size_0,
            (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
        if (moe_block_size == 8)
            max_c_tmp_size *= 2;

        c_tmp_bytes = max_c_tmp_size * sizeof(float);
    }

    // Detect groupsize and act_order
    int num_groups = -1;
    int group_size = -1;

    // b_scales.ndim = 3

    RUNTIME_CHECK(b_scales_size_2 == size_n, "b_scales dim 2 = ", b_scales_size_2,
                  " is not size_n = ", size_n);
    num_groups = b_scales_size_1;

    bool has_act_order = false;
    if (g_idx != nullptr && perm != nullptr)
    {
        // g_idx.size(-1) == size_k && perm.size(-1) == size_k

        has_act_order = true;
    }

    int a_tmp_bytes = 0;
    if (has_act_order)
    {
        a_tmp_bytes = size_m * top_k * size_k * sizeof(scalar_t);
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
                ", is not divisible by b_scales.size(1) = ", b_scales_size_1);
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = -1;
        }
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
    // b_bias = [xx, size_n]
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
    // if (has_zp)
    // {
    //     // b_zeros.ndim == 3
    //     if (is_zp_float)
    //     {
    //         RUNTIME_CHECK(b_zeros_size_2 == size_n,
    //                       "b_zeros dim 2 = ", b_zeros_size_2,
    //                       " is not size_n = ", size_n);
    //         RUNTIME_CHECK(num_groups == b_zeros_size_1,
    //                       "b_zeros dim 1 = ", b_zeros_size_1,
    //                       " is not num_groups = ", num_groups);
    //         RUNTIME_CHECK(num_groups != -1, "num_groups must be != -1");
    //     }
    //     else
    //     {
    //         RUNTIME_CHECK(b_zeros_size_1 == num_groups,
    //                       "b_zeros dim 1 = ", b_zeros_size_1,
    //                       " is not num_groups = ", num_groups);
    //         RUNTIME_CHECK(b_zeros_size_2 == size_n / pack_factor,
    //                       "b_zeros dim 2 = ", b_zeros_size_2,
    //                       " is not size_n / pack_factor = ", size_n / pack_factor);
    //     }
    // }

    // Verify workspace size
    RUNTIME_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                  "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                  MARLIN_NAMESPACE_NAME::min_thread_n);

    int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
    int min_workspace_size = min(
        max_n_tiles * (int)(sorted_token_ids_size_0 / moe_block_size), sms * 4);
    const int total_bytes = c_tmp_bytes + a_tmp_bytes + min_workspace_size;
    void *total_buffer;
    void *workspace = nullptr;
    void *c_tmp = nullptr;
    void *a_tmp = nullptr;
    if (total_bytes > 0)
    {
        cudaMalloc(&total_buffer, total_bytes);
    }
    uint8_t *ptr = reinterpret_cast<uint8_t *>(total_buffer);
    if (use_fp32_reduce && !use_atomic_add)
    {
        c_tmp = reinterpret_cast<float *>(ptr);
        ptr += c_tmp_bytes;
    }
    if (has_act_order && a_tmp_bytes > 0)
    {
        a_tmp = ptr;
        ptr += a_tmp_bytes;
    }
    if (min_workspace_size > 0)
    {
        workspace = ptr;
        cudaMemset(workspace, 0, min_workspace_size);
        ptr += min_workspace_size;
    }

    // a_scales = float; global_scale = float

    if (a_type.size_bits() == 16)
    {
        RUNTIME_CHECK(
            (a_type == c_type),
            "scalar type of a must be the same with c for 16 bit activation");
    }

    MARLIN_NAMESPACE_NAME::marlin_mm(
        a, b_q_weight, c, c_tmp,
        b_bias, a_scales, b_scales,
        global_scale, b_zeros, g_idx,
        perm, a_tmp, sorted_token_ids,
        expert_ids, num_tokens_past_padded,
        topk_weights, moe_block_size, num_experts, top_k,
        mul_topk_weights, size_m, size_n, size_k, workspace, a_type,
        b_type, c_type, s_type, has_bias, has_act_order, is_k_full, has_zp,
        num_groups, group_size, device_id, stream,
        thread_k, thread_n, sms, blocks_per_sm, use_atomic_add, use_fp32_reduce,
        is_zp_float);

    cudaFree(total_buffer);
}

enum DataType
{
    DT_FLOAT16 = 0,
    DT_BFLOAT16 = 1,
};

extern "C" void vllm_moe_wna16_marlin_gemm_nv(
    void *c,
    const void *a,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *a_scales,     // float *a_scales
    void *global_scale, // float *global_scale
    void *b_zeros,
    void *g_idx,
    void *perm,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_past_padded,
    void *topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    int64_t b_type_id,
    int size_m,
    int size_n,
    int size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int sorted_token_ids_size_0,
    int b_q_weight_size_0,
    int b_q_weight_size_1,
    int b_q_weight_size_2,
    int b_scales_size_1,
    int b_scales_size_2,
    int dataType)
{

    int64_t thread_k = -1;
    int64_t thread_n = -1;
    int64_t blocks_per_sm = -1;
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
        vllm_moe_wna16_marlin_gemm<__half, __half>(
            a, c,
            b_q_weight,
            b_bias, b_scales,
            a_scales,     // float *a_scales
            global_scale, // float *global_scale
            b_zeros,
            g_idx,
            perm,
            sorted_token_ids, expert_ids,
            num_tokens_past_padded, topk_weights,
            static_cast<int64_t>(moe_block_size), static_cast<int64_t>(top_k), static_cast<int64_t>(mul_topk_weights),
            b_type_id, static_cast<int64_t>(size_m), static_cast<int64_t>(size_n),
            static_cast<int64_t>(size_k), is_k_full, use_atomic_add, use_fp32_reduce,
            is_zp_float, thread_k, thread_n,
            blocks_per_sm,
            sorted_token_ids_size_0,
            static_cast<int64_t>(b_q_weight_size_0),
            static_cast<int64_t>(b_q_weight_size_1),
            static_cast<int64_t>(b_q_weight_size_2),
            static_cast<int64_t>(b_scales_size_1),
            static_cast<int64_t>(b_scales_size_2),
            stream);
        break;
    case DT_BFLOAT16:
        vllm_moe_wna16_marlin_gemm<__nv_bfloat16, __nv_bfloat16>(
            a, c,
            b_q_weight,
            b_bias, b_scales,
            a_scales,     // float *a_scales
            global_scale, // float *global_scale
            b_zeros,
            g_idx,
            perm,
            sorted_token_ids, expert_ids,
            num_tokens_past_padded, topk_weights,
            static_cast<int64_t>(moe_block_size), static_cast<int64_t>(top_k), static_cast<int64_t>(mul_topk_weights),
            b_type_id, static_cast<int64_t>(size_m), static_cast<int64_t>(size_n),
            static_cast<int64_t>(size_k), is_k_full, use_atomic_add, use_fp32_reduce,
            is_zp_float, thread_k, thread_n,
            blocks_per_sm,
            sorted_token_ids_size_0,
            static_cast<int64_t>(b_q_weight_size_0),
            static_cast<int64_t>(b_q_weight_size_1),
            static_cast<int64_t>(b_q_weight_size_2),
            static_cast<int64_t>(b_scales_size_1),
            static_cast<int64_t>(b_scales_size_2),
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
