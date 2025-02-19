#include "common_npu.h"
#include <iostream>

#define checkASCENDError(call)                                       \
    {                                                                \
        auto err = call;                                             \
        if (ACL_SUCCESS != err)                                      \
        {                                                            \
            fprintf(stderr, "ASCEND error in %s:%i : .\n", __FILE__, \
                    __LINE__);                                       \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do                                    \
    {                                     \
        if (!(cond))                      \
        {                                 \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    // auto ret = aclInit(nullptr);
    // CHECK_RET(ret == ACL_SUCCESS, printf("aclInit failed. ERROR: %d\n", ret); return ret);

    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
void Finalize(int32_t deviceId, aclrtStream &stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    // aclFinalize();
}
