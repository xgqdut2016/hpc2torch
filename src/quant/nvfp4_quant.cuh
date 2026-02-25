/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cuda.h>
#include <cuda_fp8.h>
#include <cutlass/arch/config.h>

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
    using Type = half2;
}; // keep for generality

template <>
struct TypeConverter<half2> {
    using Type = half;
};

template <>
struct TypeConverter<half> {
    using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
    using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
    using Type = __nv_bfloat162;
};

#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

__device__ inline uint8_t fp32_to_e2m1_function(float f) {
    uint32_t sign = 0;
    if (f < 0) {
        sign = 0x8;
        f = -f;
    }

    // 处理特殊值
    if (isinf(f)) {
        return sign | 0x7;
    }
    if (isnan(f)) {
        return sign | 0xF;
    }

    // 快速近似转换
    f = fminf(f, 6.0f);

    // 通过比较直接映射到最近的E2M1值
    if (f < 0.25f) {
        return sign | 0x0; // 0
    }
    if (f < 0.75f) {
        return sign | 0x1; // 0.5
    }
    if (f < 1.25f) {
        return sign | 0x2; // 1.0
    }
    if (f < 1.75f) {
        return sign | 0x3; // 1.5
    }
    if (f < 2.5f) {
        return sign | 0x4; // 2.0
    }
    if (f < 3.5f) {
        return sign | 0x5; // 3.0
    }
    if (f < 5.0f) {
        return sign | 0x6; // 4.0
    }
    return sign | 0x7; // 6.0
}

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
    // PTX instructions used here requires >= sm100f.
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED || CUTLASS_ARCH_MMA_SM120A_ENABLED || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ > 1000))
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]),
          "f"(array[1]),
          "f"(array[2]),
          "f"(array[3]),
          "f"(array[4]),
          "f"(array[5]),
          "f"(array[6]),
          "f"(array[7]));
    return val;
#else
    uint32_t val = 0;

#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t e2m1_val = fp32_to_e2m1_function(array[i]);
        val |= (e2m1_val << (i * 4));
    }

    return val;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
    // PTX instructions used here requires >= sm100f.
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED || CUTLASS_ARCH_MMA_SM120A_ENABLED || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ > 1000))
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x),
          "f"(array[0].y),
          "f"(array[1].x),
          "f"(array[1].y),
          "f"(array[2].x),
          "f"(array[2].y),
          "f"(array[3].x),
          "f"(array[3].y));
    return val;
#else
    uint32_t val = 0;

    // 处理第1个float2
    uint8_t b0 = fp32_to_e2m1_function(array[0].x);
    uint8_t b1 = fp32_to_e2m1_function(array[0].y);
    val |= ((uint32_t)((b1 << 4) | (b0 & 0xF))) << 0;

    // 处理第2个float2
    uint8_t b2 = fp32_to_e2m1_function(array[1].x);
    uint8_t b3 = fp32_to_e2m1_function(array[1].y);
    val |= ((uint32_t)((b3 << 4) | (b2 & 0xF))) << 8;

    // 处理第3个float2
    uint8_t b4 = fp32_to_e2m1_function(array[2].x);
    uint8_t b5 = fp32_to_e2m1_function(array[2].y);
    val |= ((uint32_t)((b5 << 4) | (b4 & 0xF))) << 16;

    // 处理第4个float2
    uint8_t b6 = fp32_to_e2m1_function(array[3].x);
    uint8_t b7 = fp32_to_e2m1_function(array[3].y);
    val |= ((uint32_t)((b7 << 4) | (b6 & 0xF))) << 24;

    return val;
#endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t *cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numCols, SFType *SFout) {

    static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

    // One pair of threads write one SF to global memory.
    // TODO: stage through smem for packed STG.32
    // is it better than STG.8 from 4 threads ?
    if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
        // SF vector index (16 elements share one SF in the K dimension).
        int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
        int32_t mIdx = rowIdx;

        // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
        // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

        int32_t mTileIdx = mIdx / (32 * 4);
        // SF vector size 16.
        int factor = CVT_FP4_SF_VEC_SIZE * 4;
        int32_t numKTiles = (numCols + factor - 1) / factor;
        int64_t mTileStride = numKTiles * 32 * 4 * 4;

        int32_t kTileIdx = (kIdx / 4);
        int64_t kTileStride = 32 * 4 * 4;

        // M tile layout [32, 4] is column-major.
        int32_t outerMIdx = (mIdx % 32);
        int64_t outerMStride = 4 * 4;

        int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
        int64_t innerMStride = 4;

        int32_t innerKIdx = (kIdx % 4);
        int64_t innerKStride = 1;

        // Compute the global offset.
        int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride + innerKIdx * innerKStride;

        return reinterpret_cast<uint8_t *>(SFout) + SFOffset;
    }
    return nullptr;
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
    typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
    __nv_fp8x2_e4m3 elts[8];
};

