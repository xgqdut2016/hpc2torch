#include <stdio.h>
#include <math.h>
#include "cpu/common_cpu.h"
template <typename T>
void layernorm_cpu(void const *input, void const *scale, void const *bias, void *output, float eps, int size, int behindsize)
{
    int frontsize = size / behindsize;
    auto source = reinterpret_cast<const T *>(input);
    auto weight = reinterpret_cast<const T *>(scale);
    auto _bias = reinterpret_cast<const T *>(bias);
    auto destination = reinterpret_cast<T *>(output);
    if (sizeof(T) == 4)
    {
        for (int i = 0; i < frontsize; i++)
        {
            int tid = i * behindsize;
            float mu = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                mu += source[tid + id];
            }
            mu /= behindsize;
            float sigma2Partial = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                sigma2Partial += (source[tid + id] - mu) * (source[tid + id] - mu);
            }
            float sigma2 = 1.0f / sqrt(sigma2Partial / behindsize + eps);
            for (int id = 0; id < behindsize; id++)
            {
                destination[tid + id] = (source[tid + id] - mu) * weight[id] * sigma2 + _bias[id];
            }
        }
    }
    else if (sizeof(T) == 2)
    {
        for (int i = 0; i < frontsize; i++)
        {
            int tid = i * behindsize;
            float mu = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                mu += f16_to_f32(source[tid + id]);
            }
            mu /= behindsize;
            float sigma2Partial = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                sigma2Partial += (f16_to_f32(source[tid + id]) - mu) * (f16_to_f32(source[tid + id]) - mu);
            }
            float sigma2 = 1.0f / sqrt(sigma2Partial / behindsize + eps);
            for (int id = 0; id < behindsize; id++)
            {
                float tmp = (f16_to_f32(source[tid + id]) - mu) * f16_to_f32(weight[id]) * sigma2 + f16_to_f32(_bias[id]);
                destination[tid + id] = f32_to_f16(tmp);
            }
        }
    }
}
extern "C" void layernorm_cpu_f32(void const *input, void const *scale, void const *bias, void *output, float eps, int size, int behindsize)
{
    layernorm_cpu<float>(input, scale, bias, output, eps, size, behindsize);
}
extern "C" void layernorm_cpu_f16(void const *input, void const *scale, void const *bias, void *output, float eps, int size, int behindsize)
{
    layernorm_cpu<uint16_t>(input, scale, bias, output, eps, size, behindsize);
}