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

def gatherElements(input, index, axis=0):
    return torch.gather(input, axis, index.to(torch.int64))
def test(inputShape, indexShape, axis, device):
    byteSize = 2
    if byteSize == 2:
        test_dtype = torch.float16
    elif byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)
    indexTensor = torch.randint(0, inputShape[axis], indexShape, device=device, dtype=torch.int32)
    if(device == "cuda"):#有些国产平台比如说MLU不支持int64计算
        indexTensor = indexTensor.to(int64)
    

    outputShape = indexShape
    Q_output = torch.zeros(outputShape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    ndim = len(inputShape)
    x_array = np.array(inputShape, dtype=np.int32)
    x_shape = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    w_array = np.array(indexShape, dtype=np.int32)
    w_shape = w_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    y_array = np.array(outputShape, dtype=np.int32)
    y_shape = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_gatherElements_time = performance.BangProfile((gatherElements, (inputTensor, indexTensor, axis)))
        lib.gatherElements_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_gatherElements_time = performance.BangProfile((lib.gatherElements_cnnl, 
        (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, ndim, axis, byteSize))) 
    elif device == "npu":
        torch_gatherElements_time = performance.AscendProfile((gatherElements, (inputTensor, indexTensor, axis)))
        lib.gatherElements_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_gatherElements_time = performance.AscendProfile((lib.gatherElements_aclnn, 
        (input_ptr, index_ptr, output_ptr, x_shape, w_shape, y_shape, ndim, axis, byteSize))) 
    performance.logBenchmark(torch_gatherElements_time, custom_gatherElements_time)

    tmpa = gatherElements(inputTensor, indexTensor, axis).to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test gatherElements on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0),
        ((3, 2), (1, 2), 1),
        ((50257, 1024), (16, 768), 0),#torch.gather要求inputshape[d] >= indexShape[d] for all d != axis
        ((9, 9, 10, 9), (16, 9, 10, 9), 0),
        ((9, 9, 10, 9), (9, 16, 10, 9), 1),
        ((9, 9, 10, 9), (9, 9, 16, 9), 2),
        ((9, 9, 10, 9), (9, 9, 10, 16), 3),

         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for inputShape , indexShape, axis in test_cases:
    test(inputShape , indexShape, axis, args.device)
    