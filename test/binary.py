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
def add(x, y):
    return torch.add(x, y)
def mul(x, y):
    return torch.mul(x, y)
def div(x, y):
    return torch.div(x, y)
def maximum(x, y):
    return torch.max(x, y)
def min(x, y):
    return torch.min(x, y)
def pow(x, y):
    return torch.pow(x, y)
def mod(x, y):
    return torch.fmod(x, y)
def bitwiseAnd(x, y):
    return torch.bitwise_and(x, y)
def bitwiseOr(x, y):
    return torch.bitwise_or(x, y)
def bitwiseNot(x, y):
    return torch.bitwise_not(x, y)
def bitwisexor(x, y):
    return torch.bitwise_xor(x, y)
def test(c_shape, a_shape, b_shape, device):
    operators = ["max", "min", "add", "pow", "div", "mul", "bitwiseOr"]
    operator = operators[2]
    byteSize = 4
    if (operator[:3] == "bit"):
        if byteSize == 2:
            tensor_dtype = torch.int16
        elif byteSize == 4:
            tensor_dtype = torch.int32
    else:
        if byteSize == 2:
            tensor_dtype = torch.float16
        elif byteSize == 4:
            tensor_dtype = torch.float32
    print(
        f"Testing {operator} on {device} with aShape:{a_shape}, bShape:{b_shape}, cShape:{c_shape}, dtype:{tensor_dtype}"
    )
    
    if tensor_dtype == torch.int16 or tensor_dtype == torch.int32:
        a = torch.ones(a_shape, dtype=tensor_dtype).to(device)
        b = 2 * torch.ones(b_shape, dtype=tensor_dtype).to(device)
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(device)
    else:
        a = torch.rand(a_shape, dtype=tensor_dtype).to(device)
        b = torch.rand(b_shape, dtype=tensor_dtype).to(device)
        c = torch.zeros(c_shape, dtype=tensor_dtype).to(device)
    aDim = len(a_shape)
    bDim = len(b_shape)
    cDim = len(c_shape)
    aData = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bData = ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cData = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    aShape = np.array(a_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bShape = np.array(b_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(c_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if operator == "div":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((div, (a, b)))  # 可以替换为mul, div
            lib.div_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.div_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((div, (a, b)))  # 可以替换为mul, div
            lib.div_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.div_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                           
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = div(a, b).to('cpu').numpy().flatten()
    elif operator == "mul":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((mul, (a, b)))  # 可以替换为mul, div
            lib.mul_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.mul_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((mul, (a, b)))  # 可以替换为mul, div
            lib.mul_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.mul_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))     
        if device == "cuda":
            torch_elementwise_time = performance.CudaProfile((mul, (a, b)))  # 可以替换为mul, div
            lib.mul_cudnn.argtypes = [
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
            custom_elementwise_time = \
            performance.CudaProfile((lib.mul_cudnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                           
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = mul(a, b).to('cpu').numpy().flatten()
    elif operator == "add":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((add, (a, b)))  # 可以替换为mul, div
            lib.add_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.add_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((add, (a, b)))  # 可以替换为mul, div
            lib.add_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.add_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))  
        if device == "cuda":
            torch_elementwise_time = performance.CudaProfile((add, (a, b)))  # 可以替换为mul, div
            lib.add_cudnn.argtypes = [
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
            custom_elementwise_time = \
            performance.CudaProfile((lib.add_cudnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                           
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = add(a, b).to('cpu').numpy().flatten()
    elif operator == "max":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((maximum, (a, b)))  # 可以替换为mul, div
            lib.max_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.max_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((maximum, (a, b)))  # 可以替换为mul, div
            lib.max_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.max_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))    
        if device == "cuda":
            torch_elementwise_time = performance.CudaProfile((maximum, (a, b)))  # 可以替换为mul, div
            lib.max_cudnn.argtypes = [
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
            custom_elementwise_time = \
            performance.CudaProfile((lib.max_cudnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                         
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = maximum(a, b).to('cpu').numpy().flatten()
    elif operator == "min":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((min, (a, b)))  # 可以替换为mul, div
            lib.min_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.min_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((min, (a, b)))  # 可以替换为mul, div
            lib.min_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.min_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))  
        if device == "cuda":
            torch_elementwise_time = performance.CudaProfile((min, (a, b)))  # 可以替换为mul, div
            lib.min_cudnn.argtypes = [
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
            custom_elementwise_time = \
            performance.CudaProfile((lib.min_cudnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                           
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = min(a, b).to('cpu').numpy().flatten()
    elif operator == "pow":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((pow, (a, b)))  # 可以替换为mul, div
            lib.pow_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.pow_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        if device == "npu":
            torch_elementwise_time = performance.AscendProfile((pow, (a, b)))  # 可以替换为mul, div
            lib.pow_aclnn.argtypes = [
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
            custom_elementwise_time = \
            performance.AscendProfile((lib.pow_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))                           
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = pow(a, b).to('cpu').numpy().flatten()
    elif operator == "mod":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((mod, (a, b)))  # 可以替换为mul, div
            lib.mod_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.mod_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = mod(a, b).to('cpu').numpy().flatten()
    elif operator == "bitwiseAnd":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((bitwiseAnd, (a, b)))  # 可以替换为mul, div
            lib.bitwiseAnd_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.bitwiseAnd_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = bitwiseAnd(a, b).to('cpu').numpy().flatten()
    elif operator == "bitwiseOr":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((bitwiseOr, (a, b)))  # 可以替换为mul, div
            lib.bitwiseOr_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.bitwiseOr_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = bitwiseOr(a, b).to('cpu').numpy().flatten()
    elif operator == "bitwiseNot":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((bitwiseNot, (a, b)))  # 可以替换为mul, div
            lib.bitwiseNot_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.bitwiseNot_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = bitwiseNot(a, b).to('cpu').numpy().flatten()
    elif operator == "bitwiseXor":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((bitwiseXor, (a, b)))  # 可以替换为mul, div
            lib.bitwiseXor_cnnl.argtypes = [
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
            custom_elementwise_time = \
            performance.BangProfile((lib.bitwiseXor_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, bDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = bitwiseXor(a, b).to('cpu').numpy().flatten()
    tmpb = c.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test elementwise on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # c_shape, a_shape, b_shape
        # ((1, 3), (1, 3), (1, 3)),
        # ((2, 4, 3), (2, 1, 3), (4, 3)),
        # ((2, 3, 4, 5), (2, 3, 4, 5), (5,)),

        # ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1)),
        # ((3, 20, 33), (3, 20, 33), (3, 20, 33)),
        # ((32, 3, 112, 112), (32, 3, 112, 112), (32, 3, 112, 112)),

        ((12, 512, 4096), (12, 512, 4096), (12, 512, 4096)),
        ((1, 512, 4096), (1, 512, 4096), (1, 512, 4096)),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for c_shape, a_shape, b_shape in test_cases:
    test(c_shape, a_shape, b_shape, args.device)