#include "acl/acl.h"
#include "aclnnop/aclnn_resize.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"
struct ResizeMode
{
    enum Mode
    {
        // Arithmetic operations:
        Nearest,
        Bilinear,

        Count, ///< Number of resize operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined resize operations.
    static const size_t numResizeMode = Count;
};

template <typename T>
void resizeAclnnDevice(void *input, float const *scaleData, void *output,
                       int *x_shape, int *y_shape,
                       int ndim,
                       ResizeMode::Mode mode,
                       aclrtStream &stream)
{
    aclDataType dataType;
    if (sizeof(T) == 2)
    {
        dataType = aclDataType::ACL_FLOAT16;
    }
    else if (sizeof(T) == 4)
    {
        dataType = aclDataType::ACL_FLOAT;
    }
    aclFormat format = aclFormat::ACL_FORMAT_NCHW;
    const char *aclnnMode;
    if (mode == ResizeMode::Nearest)
    {
        aclnnMode = "nerest";
    }
    else if (mode == ResizeMode::Bilinear)
    {
        aclnnMode = "bilinear";
    }

    std::vector<int64_t> inputDim(ndim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1); // 初始化为1
    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);

    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        outputDim[i] = int64_t(y_shape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }

    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor

    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);

    aclFloatArray *scales = nullptr;
    scales = aclCreateFloatArray(scaleData, 1); // scaleData长度=ndim

    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnResizeGetWorkspaceSize(inputTensor, scales, aclnnMode, outputTensor,
                                           &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnResizeGetWorkspaceSize failed. ERROR: %d\n", ret);
    }
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (ret != ACL_SUCCESS)
        {
            printf("allocate workspace failed. ERROR: %d\n", ret);
        }
    }

    ret = aclnnResize(workspaceAddr, workspaceSize, executor,
                      stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnResize failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyFloatArray(scales);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void resizeAclnn(void *input, float const *scaleData, void *output,
                 int *x_shape, int *y_shape,
                 int ndim,
                 ResizeMode::Mode mode)
{
    // static int count = 0;
    // printf("count is %d \n", count);
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    resizeAclnnDevice<T>(input, scaleData, output,
                         x_shape, y_shape,
                         ndim,
                         mode, stream);
    Finalize(deviceId, stream);
}
extern "C" void resize_aclnn(void *input, float const *scaleData, void *output,
                             int *x_shape, int *y_shape,
                             int ndim,
                             ResizeMode::Mode mode, int byteSize)
{
    if (byteSize == 4)
    {
        resizeAclnn<float>(input, scaleData, output,
                           x_shape, y_shape,
                           ndim,
                           mode);
    }
    else if (byteSize == 2)
    {
        resizeAclnn<uint16_t>(input, scaleData, output,
                              x_shape, y_shape,
                              ndim,
                              mode);
    }
}
