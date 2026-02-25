#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <cuda_fp8.h>

#include "../../nvfp4_quant.cuh"
#include "../../utils.h"

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type> &vec, float SFScaleVal, uint8_t *SFout)
{

    // Get absolute maximum values among the local 8 values.
    auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
    for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++)
    {
        localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }

    // Get the absolute maximum among all 16 values (two threads).
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    // Get the final absolute maximum values.
    float vecMax = float(__hmax(localMax.x, localMax.y));

    // Get the SF (max value of the vector / max value of e2m1).
    // maximum value of e2m1 = 6.0.
    // TODO: use half as compute data type.
    float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    // 8 bits representation of the SF.
    uint8_t fp8SFVal;
    // Write the SF to global memory (STG.8).
    if (UE8M0_SF)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

        __nv_fp8_e8m0 tmp;
        tmp.__x = __nv_cvt_float_to_e8m0(
            SFValue,
            __NV_SATFINITE,
            cudaRoundPosInf);

        SFValue = static_cast<float>(tmp);
        fp8SFVal = tmp.__x;

#else
        // 非 Hopper 架构，运行时 fallback
        printf("Warning: UE8M0_SF requested but not supported "
               "on this architecture. Falling back to E4M3.\n");

        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        fp8SFVal = tmp.__x;
        SFValue = static_cast<float>(tmp);
#endif
    }
    else
    {
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        fp8SFVal = tmp.__x;
        SFValue = static_cast<float>(tmp);
    }
    // Get the output scale.
    // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
    //                       reciprocal(SFScaleVal))
    float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

    if (SFout)
    {
        // Write the SF to global memory (STG.8).
        *SFout = fp8SFVal;
    }

    // Convert the input to float.
    float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
    for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++)
    {
        if constexpr (std::is_same_v<Type, half>)
        {
            fp2Vals[i] = __half22float2(vec.elts[i]);
        }
        else
        {
            fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        }
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    // Convert to e2m1 values.
    uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

    // Write the e2m1 values to global memory.
    return e2m1Vec;
}
