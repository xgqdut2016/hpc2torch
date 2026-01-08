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
def test(inputShape, indexShape, axis, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    if(device != "cuda"):#有些国产平台比如说MLU不支持int64计算
        indexTensor = torch.from_numpy(index).to(torch.int32).to(device)
    else:
        indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)#
    indsize = 1
    for i in range(len(indexShape)):
        indsize *= indexShape[i]
    frontsize = 1
    dimsize = inputShape[axis]
    behindsize = 1
    for i in range(len(inputShape)):
        if(i < axis):
            frontsize *= inputShape[i]
        elif (i > axis):
            behindsize *= inputShape[i]
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

    
    if device == "cuda":
        torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
        lib.gather_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_gather_time = performance.CudaProfile((lib.gather_nv, 
        (input_ptr, index_ptr, output_ptr, frontsize, dimsize, behindsize, indsize, byteSize)))
    elif device == "mlu":
        torch_gather_time = performance.BangProfile((gather, (rank, axis, inputTensor, indexTensor)))
        lib.gather_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_gather_time = performance.BangProfile((lib.gather_cnnl, 
        (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, x_ndim, w_ndim, y_ndim, axis, byteSize)))  
    elif device == "npu":
        torch_gather_time = performance.AscendProfile((gather, (rank, axis, inputTensor, indexTensor)))
        lib.gather_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_gather_time = performance.AscendProfile((lib.gather_aclnn, 
        (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, x_ndim, w_ndim, y_ndim, axis, byteSize)))  
    
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test gather on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis
        ((3, 2), (2, 2), 0),
        ((3, 2), (1, 2), 1),
        ((50257, 768), (16, 1024), 0),
        ((9, 9, 10, 9), (16, 10), 0),
        ((9, 9, 10, 9), (16, 10), 1),
        ((9, 9, 10, 9), (16, 10), 2),
        ((9, 9, 10, 9), (16, 10), 3),
         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for inputShape , indexShape, axis in test_cases:
    test(inputShape , indexShape, axis, args.device)
    