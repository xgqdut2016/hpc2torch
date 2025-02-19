#include <stdio.h>
#include <math.h>
#include "cpu/common_cpu.h"
template <typename T>
void softmax_cpu(void const *input, void *output, int size, int dimsize, int stride)
{
    int othersize = size / dimsize;
    auto source = reinterpret_cast<const T *>(input);
    auto destination = reinterpret_cast<T *>(output);
    if (sizeof(T) == 4)
    {

        // 假设[I, J, K, S], axis = 1, othersize = IKS
        for (int ind = 0; ind < othersize; ind++)
        {                                                            // ind = i(KS) + k(S) + s
            int tid = ind % stride + (ind - ind % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;
            float localM = -__FLT_MAX__;
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                localM = fmax(localM, source[index]);
            }
            float localS = 0.0f;
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                localS += exp(source[index] - localM);
            }
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                destination[index] = exp(source[index] - localM) / localS;
            }
        }
    }
    else if (sizeof(T) == 2)
    {
        // 假设[I, J, K, S], axis = 1, othersize = IKS
        for (int ind = 0; ind < othersize; ind++)
        {                                                            // ind = i(KS) + k(S) + s
            int tid = ind % stride + (ind - ind % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;
            float localM = -__FLT_MAX__;
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                localM = fmax(localM, f16_to_f32(source[index]));
            }
            float localS = 0.0f;
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                localS += std::exp(f16_to_f32(source[index]) - localM);
            }
            for (int j = 0; j < dimsize; j++)
            {
                int index = tid + j * stride;
                destination[index] = f32_to_f16(std::exp(f16_to_f32(source[index]) - localM) / localS);
            }
        }
    }
}
extern "C" void softmax_cpu_f32(void const *input, void *output, int size, int dimsize, int stride)
{
    softmax_cpu<float>(input, output, size, dimsize, stride);
}
extern "C" void softmax_cpu_f16(void const *input, void *output, int size, int dimsize, int stride)
{
    softmax_cpu<uint16_t>(input, output, size, dimsize, stride);
}