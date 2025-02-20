import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def maxPool(x, k, padding, stride, dilation = 1):
    pooling_layers = {
        1: torch.nn.MaxPool1d,
        2: torch.nn.MaxPool2d,
        3: torch.nn.MaxPool3d,
    }

    ndim = len(x.shape) - 2
    if ndim not in pooling_layers:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None

    ans = pooling_layers[ndim](k, stride=stride, padding=padding, dilation=dilation)(x)
    
    return ans
def avgPool(x, k, padding, stride, dilation = 1):
    pooling_layers = {
        1: torch.nn.AvgPool1d,
        2: torch.nn.AvgPool2d,
        3: torch.nn.AvgPool3d,
    }

    ndim = len(x.shape) - 2
    if ndim not in pooling_layers:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None

    if ndim == 3 and x.dtype == torch.float16:
        ans = pooling_layers[ndim](k, stride=stride, padding=padding)(x.to(torch.float32)).to(torch.float16)
    else:
        ans = pooling_layers[ndim](k, stride=stride, padding=padding)(x)
    
    return ans


def inferShape(x_shape, kernel_shape, padding, stride):
    assert (
        len(x_shape) - 2 == len(kernel_shape) == len(padding) == len(stride)
    ), "kernel, pads, and stride should have the same length; the length of input x should be 2 more than that of kernel"
    input_shape = x_shape[2:]
    output_shape = []

    for dim, k, p, s in zip(input_shape, kernel_shape, padding, stride):
        output_dim = (dim + 2 * p - k) // s + 1
        output_shape.append(output_dim)

    return x_shape[:2] + tuple(output_shape)

def test(x_shape, k_shape, padding, stride, device):
    byteSize = 2
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    # operator = "max"
    operator = "avg"
    print(
        f"Testing {operator} Pool on {device} with x_shape:{x_shape} kernel_shape:{k_shape} padding:{padding} stride:{stride} dtype:{tensor_dtype}"
    )
    nDim = len(x_shape)
    x = torch.rand(x_shape, dtype=tensor_dtype).to(device)
    y = torch.rand(inferShape(x_shape, k_shape, padding, stride), dtype=tensor_dtype).to(device)
    y_shape = []
    for i in range(len(y.shape)):
        y_shape.append(y.shape[i])

    xData = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    yData = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    xShape = np.array(x_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    yShape = np.array(y_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    p_array = np.array(padding, dtype=np.int32)
    pData = p_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    s_array = np.array(stride, dtype=np.int32)
    sData = s_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    k_array = np.array(k_shape, dtype=np.int32)#这个就是windows
    kData = k_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    dilations = []
    if nDim == 3:
        dilations = [1, ]
    elif nDim == 4:
        dilations = [1, 1]
    elif nDim == 5:
        dilations = [1, 1, 1]
    d_array = np.array(dilations, dtype=np.int32)
    dData = d_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if operator == "max":
        if device == "mlu":
            torch_Pooling_time = performance.BangProfile((maxPool, (x, k_shape, padding, stride)))
            lib.MaxPooling_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_Pooling_time = \
            performance.BangProfile((lib.MaxPooling_cnnl, (xData, yData, kData, pData, sData, dData, xShape, yShape, nDim, byteSize)))
            performance.logBenchmark(torch_Pooling_time, custom_Pooling_time)
        elif device == "npu":
            torch_Pooling_time = performance.AscendProfile((maxPool, (x, k_shape, padding, stride)))
            lib.MaxPooling_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_Pooling_time = \
            performance.AscendProfile((lib.MaxPooling_aclnn, (xData, yData, kData, pData, sData, dData, xShape, yShape, nDim, byteSize)))
            performance.logBenchmark(torch_Pooling_time, custom_Pooling_time)
        tmpa = maxPool(x, k_shape, padding, stride).to('cpu').numpy().flatten()
    elif operator == "avg":
        if device == "mlu":
            torch_Pooling_time = performance.BangProfile((avgPool, (x, k_shape, padding, stride)))
            lib.AvgPooling_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_Pooling_time = \
            performance.BangProfile((lib.AvgPooling_cnnl, (xData, yData, kData, pData, sData, dData, xShape, yShape, nDim, byteSize)))
            performance.logBenchmark(torch_Pooling_time, custom_Pooling_time)
        elif device == "npu":
            torch_Pooling_time = performance.AscendProfile((avgPool, (x, k_shape, padding, stride)))
            lib.AvgPooling_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_Pooling_time = \
            performance.AscendProfile((lib.AvgPooling_aclnn, (xData, yData, kData, pData, sData, dData, xShape, yShape, nDim, byteSize)))
            performance.logBenchmark(torch_Pooling_time, custom_Pooling_time)
        tmpa = avgPool(x, k_shape, padding, stride).to('cpu').numpy().flatten()
    tmpb = y.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test Pooling on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # x_shape, kernel_shape, padding, strides
        ((1, 1, 10), (3,), (1,), (1,)), 
        ((32, 3, 224, 224), (3, 3), (1, 1), (2, 2)),
        # ((1, 1, 16, 16, 16), (5, 5, 5), (2, 2, 2), (2, 2, 2)), #昇腾不支持5维
        # ((32, 128, 16, 16, 16), (5, 5, 5), (2, 2, 2), (2, 2, 2)),
        
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, kernel_shape, padding, strides in test_cases:
    test(x_shape, kernel_shape, padding, strides, args.device)