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
def whereFunction(aData, bData, cData):
    return torch.where(cData, aData, bData)
def broadcast(xShape, yShape):
    rankX = len(xShape)
    rankY = len(yShape)
    rank = max(rankX, rankY)
    if rank == 0:
        return ()
    else:
        if rankX < rank:
            for i in range(rank - rankX):
                xShape = (1,) + xShape
        if rankY < rank:
            for i in range(rank - rankY):
                yShape = (1,) + yShape
        outShape = ()
        for i in range(rank):
            assert (xShape[i] == yShape[i] or xShape[i] == 1 or yShape[i] == 1)
            outShape = outShape + (max(xShape[i], yShape[i]), )
        return outShape
def inferShape(xShape, yShape, zShape):
    retXY = broadcast(xShape, yShape)
    outputShape = broadcast(retXY, zShape)
    return outputShape

def test(xShape, yShape, zShape, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing where on {device} with xShape:{xShape}, yShape:{yShape}, zShape:{zShape}, dtype:{test_dtype}"
    )
    
    ndim = len(xShape)
    a = torch.rand(xShape, device=device, dtype=test_dtype, requires_grad=False)
    b = torch.rand(yShape, device=device, dtype=test_dtype, requires_grad=False)
    ctmp = torch.randint(0, 2, zShape, device=device, dtype=torch.int32, requires_grad=False)
    c = (ctmp > 0).to(device)#测试不要使用0,1元素，因为torch.int32和torch.bool字节数不一致，传入C代码很可能报错
    outputShape = inferShape(xShape, yShape, zShape)
    output = torch.zeros(outputShape, device=device, dtype=test_dtype, requires_grad=False)

    aDim = len(xShape)
    bDim = len(yShape)
    cDim = len(zShape)
    dDim = len(outputShape)

    a_ptr = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_ptr = ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    c_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    aShape = np.array(xShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bShape = np.array(yShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(zShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dShape = np.array(outputShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_where_time = performance.BangProfile((whereFunction, (a, b, c)))  # 以毫秒为单位
        lib.where_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_where_time = \
        performance.BangProfile((lib.where_cnnl, (a_ptr, b_ptr, c_ptr, output_ptr, 
        aShape, bShape, cShape, dShape,
        aDim, bDim, cDim, dDim, byteSize)))
    elif device == "npu":
        torch_where_time = performance.AscendProfile((whereFunction, (a, b, c)))  # 以毫秒为单位
        lib.where_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_where_time = \
        performance.AscendProfile((lib.where_aclnn, (a_ptr, b_ptr, c_ptr, output_ptr, 
        aShape, bShape, cShape, dShape,
        aDim, bDim, cDim, dDim, byteSize)))
    
    performance.logBenchmark(torch_where_time, custom_where_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = whereFunction(a, b, c).to('cpu').numpy().flatten()
    tmpb = output.to('cpu').numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test where on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # xShape, yShape, zShape
        ((7,), (7, ), (7,)),
        ((700, 1, 24), (700, 1200, 24), (700, 1, 1)),
        ((2, 3, 4, 5), (), (2, 3, 1, 1)),
        ((2, 3, 4, 5), (5, ), (2, 3, 4, 1)),
        ((1, 4, 5), (2, 3, 1, 1), (2, 3, 4, 1)),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for xShape, yShape, zShape in test_cases:
    test(xShape, yShape, zShape, args.device)
