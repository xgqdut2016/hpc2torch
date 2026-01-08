import torch
import ctypes
import numpy as np
from functools import partial
import argparse

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor
def extend(inputTensor, axis, num):
    rank = len(inputTensor.shape)
    N = inputTensor.shape[axis]
    x = torch.arange(0, N)
    indexTensor = x.repeat(num + 1).to(inputTensor.device)
    return gather(rank, axis, inputTensor, indexTensor)
def test(inputShape, axis, num, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Extend on {device} with x_shape:{inputShape} , axis:{axis} ,num:{num}, dtype:{test_dtype}"
    )
    ndim = len(inputShape)
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)
    
    frontsize = 1
    outputShape = list(inputShape)
    for i in range(len(inputShape)):
        if i < axis:
            frontsize *= inputShape[i]
    outputShape[axis] = (num + 1) * inputShape[axis]
    outTensor = torch.zeros(outputShape, device=device, dtype=test_dtype)

    dimsize = inputShape[axis]
    stride = inputTensor.stride()[axis]
    

    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(outTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    x_array = np.array(inputShape, dtype=np.int32)
    x_ndim = len(inputShape)
    x_shape = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    
    y_array = np.array(outputShape, dtype=np.int32)
    y_ndim = len(outputShape)
    y_shape = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    
    if device == "cuda":
        torch_extend_time = performance.CudaProfile((extend, (inputTensor, axis, num)))
        lib.extend_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
    
        custom_extend_time = performance.CudaProfile((lib.extend_nv, 
        (input_ptr, output_ptr, num, frontsize, dimsize, stride, byteSize)))
    elif device == "mlu":
        torch_extend_time = performance.BangProfile((extend, (inputTensor, axis, num)))
        lib.extend_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
    
        custom_extend_time = performance.BangProfile((lib.extend_cnnl, 
        (input_ptr, output_ptr, x_shape, y_shape, ndim, num, axis, byteSize)))
    elif device == "npu":
        torch_extend_time = performance.AscendProfile((extend, (inputTensor, axis, num)))
        lib.extend_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
    
        custom_extend_time = performance.AscendProfile((lib.extend_aclnn, 
        (input_ptr, output_ptr, x_shape, y_shape, ndim, num, axis, byteSize)))
    
    performance.logBenchmark(torch_extend_time, custom_extend_time)
    # print(inputTensor[:,0], extend(inputTensor, axis, num)[:,0] , outTensor[:,0])
    # print(inputTensor[:,1])
    tmpa = extend(inputTensor, axis, num).to('cpu').numpy().flatten()
    tmpb = outTensor.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test extend on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , axis. num
        ((2, 3, 2, 2), 1, 1),
        ((3, 2), 1, 2),
        ((50, 768, 2), 0, 2),
        ((50, 768, 2), 1, 2),
        ((50, 768, 2), 2, 2),
        ((9, 9, 10, 9), 0, 4),
        ((9, 9, 10, 9), 1, 4),
        ((9, 9, 10, 9), 2, 4),
        ((9, 9, 10, 9), 3, 4),
         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for inputShape , indexShape, axis in test_cases:
    test(inputShape , indexShape, axis, args.device)
    