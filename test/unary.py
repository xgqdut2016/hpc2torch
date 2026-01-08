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

def test(c_shape, a_shape, device):
    byteSize = 2
    
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    operators = ["hardsigmoid", "leakyrelu", "round", "prelu", "relu", "round", "sigmoid", "hardswish"]

    operator = operators[7]
    print(
        f"Testing {operator} on {device} with aShape:{a_shape}, cShape:{c_shape}, dtype:{tensor_dtype}"
    )
    
    
    a = torch.rand(a_shape, dtype=tensor_dtype).to(device)
    c = torch.zeros(c_shape, dtype=tensor_dtype).to(device)

    aDim = len(a_shape)
    cDim = len(c_shape)

    aData = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cData = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    aShape = np.array(a_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(c_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if operator == "relu":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.relu, (a, )))  # 可以替换为pRelu
            lib.relu_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.relu_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.relu, (a, )))  # 可以替换为pRelu
            lib.relu_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.relu_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.relu(a).to('cpu').numpy().flatten()
    elif operator == "gelu":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.nn.functional.gelu, (a, )))  # 可以替换为pRelu
            lib.gelu_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.gelu_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.nn.functional.gelu, (a, )))  # 可以替换为pRelu
            lib.gelu_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.gelu_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))                        
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.nn.functional.gelu(a).to('cpu').numpy().flatten()
    elif operator == "sigmoid":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.sigmoid, (a, )))  # 可以替换为pRelu
            lib.sigmoid_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.sigmoid_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.sigmoid, (a, )))  # 可以替换为pRelu
            lib.sigmoid_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.sigmoid_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize))) 
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.sigmoid(a).to('cpu').numpy().flatten()
    elif operator == "hardswish":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.nn.functional.hardswish, (a, )))  # 可以替换为pRelu
            lib.hardswish_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.hardswish_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.nn.functional.hardswish, (a, )))  # 可以替换为pRelu
            lib.hardswish_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.hardswish_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize))) 
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.nn.functional.hardswish(a).to('cpu').numpy().flatten()
    elif operator == "hardsigmoid":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.nn.functional.hardsigmoid, (a, )))  # 可以替换为pRelu
            lib.hardsigmoid_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.hardsigmoid_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.nn.functional.hardsigmoid, (a, )))  # 可以替换为pRelu
            lib.hardsigmoid_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.hardsigmoid_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))                             
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.nn.functional.hardsigmoid(a).to('cpu').numpy().flatten()
    elif operator == "leakyrelu":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.nn.functional.leaky_relu, (a, )))  # 可以替换为pRelu
            lib.leakyRelu_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.leakyRelu_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.nn.functional.leaky_relu, (a, )))  # 可以替换为pRelu
            lib.leakyRelu_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.leakyRelu_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))                        
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.nn.functional.leaky_relu(a).to('cpu').numpy().flatten()
    elif operator == "round":
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.round, (a, )))  # 可以替换为pRelu
            lib.round_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.BangProfile((lib.round_cnnl, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.round, (a, )))  # 可以替换为pRelu
            lib.round_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_elementwise_time = \
            performance.AscendProfile((lib.round_aclnn, (aData, cData, aShape, cShape,
                                    aDim, cDim, byteSize)))                              
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.round(a).to('cpu').numpy().flatten()
    elif operator == "prelu":
        if aDim == 1:
            b_shape = [1, ]
        else:
            b_shape = [a_shape[1]]
        b = torch.rand(b_shape, dtype=tensor_dtype).to(device)

        bData = ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        bShape = np.array(b_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        if device == "mlu":
            torch_elementwise_time = performance.BangProfile((torch.nn.functional.prelu, (a, b)))  # 可以替换为pRelu
            lib.pRelu_cnnl.argtypes = [
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
            performance.BangProfile((lib.pRelu_cnnl, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, 1, cDim, byteSize)))
        elif device == "npu":
            torch_elementwise_time = performance.AscendProfile((torch.nn.functional.prelu, (a, b)))  # 可以替换为pRelu
            lib.pRelu_aclnn.argtypes = [
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
            performance.AscendProfile((lib.pRelu_aclnn, (aData, bData, cData, aShape, bShape, cShape,
                                    aDim, 1, cDim, byteSize)))                            
        performance.logBenchmark(torch_elementwise_time, custom_elementwise_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = torch.nn.functional.prelu(a, b).to('cpu').numpy().flatten()
    
    tmpb = c.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test elementwise on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # c_shape, a_shape, device
        ((1000, ), (1000, )),
        ((1, 3), (1, 3)),
        ((2, 4, 3), (2, 4, 3)),
        ((2, 3, 4, 5), (2, 3, 4, 5)),
        
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for c_shape, a_shape in test_cases:
    test(c_shape, a_shape, args.device)