#ifndef __COMMON_CPU_H__
#define __COMMON_CPU_H__

#include <cmath>
#include <cstdint>

// convert half-precision float to single-precision float
float f16_to_f32(uint16_t code);

// convert single-precision float to half-precision float
uint16_t f32_to_f16(float val);

#endif // __COMMON_CPU_H__
