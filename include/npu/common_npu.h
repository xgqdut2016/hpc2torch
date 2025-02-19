#ifndef __COMMON_NPU_H__
#define __COMMON_NPU_H__

#include <cmath>
#include <cstdint>
#include "acl/acl.h"
// convert half-precision float to single-precision float
int Init(int32_t deviceId, aclrtStream *stream);

// convert single-precision float to half-precision float
void Finalize(int32_t deviceId, aclrtStream &stream);

#endif // __COMMON_NPU_H__
