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

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor
def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing Softmax on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#
    indSize = 1
    for i in range(len(indexShape)):
        indSize *= indexShape[i]
    stride = inputTensor.stride()[axis]
    othersize = 1
    for i in range(len(inputShape)):
        if(i != axis):
            othersize *= inputShape[i]
    outputShape = []
    for i in range(len(outTensor.shape)):
        outputShape.append(outTensor.shape[i])
    Q_output = torch.zeros(outputShape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    x_array = np.array(inputShape, dtype=np.int32)
    x_ndim = len(inputShape)
    x_shape = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    w_array = np.array(indexShape, dtype=np.int32)
    w_ndim = len(indexShape)
    w_shape = w_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    y_array = np.array(outputShape, dtype=np.int32)
    y_ndim = len(outputShape)
    y_shape = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if test_dtype == torch.float32:
        if device == "cuda":
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            lib.gather_nv_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_gather_time = performance.CudaProfile((lib.gather_nv_f32, (input_ptr, index_ptr, output_ptr, stride, indSize, othersize)))
        if device == "mlu":
            torch_gather_time = performance.BangProfile((gather, (rank, axis, inputTensor, indexTensor)))
            lib.gather_cnnl_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_gather_time = performance.BangProfile((lib.gather_cnnl_f32, 
            (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, x_ndim, w_ndim, y_ndim, axis)))  
    if test_dtype == torch.float16:
        if device == "cuda":
            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
            lib.gather_nv_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_gather_time = performance.CudaProfile((lib.gather_nv_f16, (input_ptr, index_ptr, output_ptr, stride, indSize, othersize))) 
        if device == "mlu":
            torch_gather_time = performance.BangProfile((gather, (rank, axis, inputTensor, indexTensor)))
            lib.gather_cnnl_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_gather_time = performance.BangProfile((lib.gather_cnnl_f16, 
            (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, x_ndim, w_ndim, y_ndim, axis)))  
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test gather on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),

        ((3, 2), (2, 2), 0, torch.float32, "mlu"),
        ((3, 2), (1, 2), 1, torch.float32, "mlu"),
        ((50257, 768), (16, 1024), 0, torch.float32, "mlu"),
        

        ((3, 2), (2, 2), 0, torch.float16, "mlu"),
        ((3, 2), (1, 2), 1, torch.float16, "mlu"),
        ((50257, 768), (16, 1024), 0, torch.float16, "mlu"),
        
         
]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)
    