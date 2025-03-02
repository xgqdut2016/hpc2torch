#include <stdio.h>
#include "cpu/common_cpu.h"
#include "omp.h"
template <typename T>
void clipDevice(void const *input, void *output, float minValue, float maxValue, int n)
{
    auto source = reinterpret_cast<const T *>(input);
    auto destination = reinterpret_cast<T *>(output);
    if (sizeof(T) == 2)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            destination[i] = (f16_to_f32(source[i]) < minValue ? f32_to_f16(minValue) : (f16_to_f32(source[i]) > maxValue ? f32_to_f16(maxValue) : source[i]));
        }
    }
    else if (sizeof(T) == 4)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            destination[i] = (source[i] < minValue ? minValue : (source[i] > maxValue ? maxValue : source[i]));
        }
    }
}

extern "C" void clip_cpu(void const *input, void *output, float minValue, float maxValue, int n, int byteSize)
{
    if (byteSize == 2)
    {
        clipDevice<uint16_t>(input, output, minValue, maxValue, n);
    }
    else if (byteSize == 4)
    {
        clipDevice<float>(input, output, minValue, maxValue, n);
    }
}
