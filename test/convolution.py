import torch
import ctypes
import torch.nn.functional as F
from functools import partial
import argparse
from typing import List, Tuple
import math

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def conv(x, w, stride, padding, dilation):
    match len(x.shape) - 2:
        case 1:
            return F.conv1d(
                x, w, stride=stride, padding=padding, dilation=dilation
            )
        case 2:
            return F.conv2d(
                x, w, stride=stride, padding=padding, dilation=dilation
            )
        case 3:
            return F.conv3d(
                x, w, stride=stride, padding=padding, dilation=dilation
            )
        case _:
            print("Error: Pytorch -> Unsupported tensor dimension")
            return None
def inferShape(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[int, ...]:
    assert (
        len(x_shape) == len(w_shape) == len(pads) + 2 == len(dilations) + 2 == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilatinos should have the same length; the length of pads should be that of x - 2"
    output_dims = [
        math.floor(
            (x_shape[i+2] + 2 * pads[i] - dilations[i] * (w_shape[i+2] - 1) - 1)
            / strides[i]
            + 1
        )
        for i in range(len(pads))
    ]
    return (x_shape[0], w_shape[0]) + tuple(output_dims)

def test(x_shape, w_shape, pads, strides, dilations, test_dtype, device):
    print(
        f"Testing Batchnorm on {device} with x_shape:{x_shape}, w_shape:{w_shape}, pads: {pads}, strides: {strides}, dilations: {dilations}, dtype:{test_dtype}"
    )           
    ndim = len(x_shape) 

    x = torch.rand(x_shape, dtype=test_dtype).to(device)
    w = torch.rand(w_shape, dtype=test_dtype).to(device)
    y_shape = inferShape(x.shape, w.shape, pads, strides, dilations)
    y = torch.zeros(y_shape, dtype=test_dtype).to(device)
    print(y_shape)
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    w_ptr = ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    import numpy as np
    x_array = np.array(x_shape, dtype=np.int32)
    xShape = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    w_array = np.array(w_shape, dtype=np.int32)
    wShape = w_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    y_array = np.array(y_shape, dtype=np.int32)
    yShape = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    p_array = np.array(pads, dtype=np.int32)
    pData = p_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    s_array = np.array(strides, dtype=np.int32)
    sData = s_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    d_array = np.array(dilations, dtype=np.int32)
    dData = d_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if test_dtype == torch.float32:
        if device == "mlu":
            torch_convolution_time = performance.BangProfile((conv, (x, w, strides, pads, dilations))) 
            lib.convolution_cnnl_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),#pads
                ctypes.POINTER(ctypes.c_int),#strides
                ctypes.POINTER(ctypes.c_int),#dilations
                ctypes.POINTER(ctypes.c_int),#x_shape
                ctypes.POINTER(ctypes.c_int),#w_shape
                ctypes.POINTER(ctypes.c_int),#y_shape
                ctypes.c_int
            ]           
            custom_convolution_time = \
            performance.BangProfile((lib.convolution_cnnl_f32, (x_ptr, w_ptr, y_ptr, pData, sData, dData, xShape, wShape, yShape, ndim)))
    if test_dtype == torch.float16:
        if device == "mlu":
            torch_convolution_time = performance.BangProfile((conv, (x, w, strides, pads, dilations))) 
            lib.convolution_cnnl_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),#pads
                ctypes.POINTER(ctypes.c_int),#strides
                ctypes.POINTER(ctypes.c_int),#dilations
                ctypes.POINTER(ctypes.c_int),#x_shape
                ctypes.POINTER(ctypes.c_int),#w_shape
                ctypes.POINTER(ctypes.c_int),#y_shape
                ctypes.c_int
            ]           
            custom_convolution_time = \
            performance.BangProfile((lib.convolution_cnnl_f16, (x_ptr, w_ptr, y_ptr, pData, sData, dData, xShape, wShape, yShape, ndim)))
    performance.logBenchmark(torch_convolution_time, custom_convolution_time)

    # 将结果转换回 PyTorch 张量以进行比较
    
    tmpa = conv(x, w, strides, pads, dilations).to('cpu').detach().numpy().flatten()
    
    tmpb = y.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test convolution on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [       
        ((32, 3, 128, 128),
            (64, 3, 5, 5),
            (2, 2),
            (2, 2),
            (1, 1), torch.float32, 'mlu'), 
        ((1, 1, 4, 4, 4),
            (1, 1, 5, 5, 5),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1), torch.float32, 'mlu'), 
        # ((32, 3, 32, 32, 32),
        #     (64, 3, 5, 5, 5),
        #     (3, 2, 2),
        #     (4, 3, 3),
        #     (2, 2, 1), torch.float32, 'mlu'),    
        # ((32, 3, 128, 128),
        #     (64, 3, 5, 5),
        #     (2, 2),
        #     (2, 2),
        #     (1, 1), torch.float16, 'mlu'), 
        # ((1, 1, 4, 4, 4),
        #     (1, 1, 5, 5, 5),
        #     (1, 1, 1),
        #     (1, 1, 1),
        #     (1, 1, 1), torch.float16, 'mlu'), 
        # ((32, 3, 32, 32, 32),
        #     (64, 3, 5, 5, 5),
        #     (3, 2, 2),
        #     (4, 3, 3),
        #     (2, 2, 1), torch.float16, 'mlu'),    
]

filtered_test_cases = [
    (x_shape, w_shape, pads, strides, dilations, test_dtype, device)
    for x_shape, w_shape, pads, strides, dilations, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for x_shape, w_shape, pads, strides, dilations, test_dtype, device in filtered_test_cases:
    test(x_shape, w_shape, pads, strides, dilations, test_dtype, device)

