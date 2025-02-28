import torch
import ctypes
import numpy as np
import torch.nn as nn
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def test(test_shape, axis, eps, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Layernorm on {device} with test_shape:{test_shape}, axis:{axis} ,dtype:{test_dtype}, eps:{eps}"
    )
    ndim = len(test_shape)
    normlize_shape = []
    for i in range(axis, ndim):
        normlize_shape.append(test_shape[i])
    size = 1
    behindsize = 1
    for i in range(ndim):
        size *= test_shape[i]
        if (i >= axis):
            behindsize *= test_shape[i]
    input = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    scale = torch.rand(normlize_shape, device=device, dtype=test_dtype, requires_grad=False)
    bias = torch.rand(normlize_shape, device=device, dtype=test_dtype, requires_grad=False)
    output = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    layer_norm = nn.LayerNorm(normlize_shape, elementwise_affine=True, eps = eps)
    layer_norm.weight.data = scale
    layer_norm.bias.data = bias

    
    np_array = np.array(test_shape, dtype=np.int32)
    ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if device == "cuda":
        torch_layernorm_time = performance.CudaProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
        lib.layernorm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_layernorm_time = \
        performance.CudaProfile((lib.layernorm_nv, (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize, byteSize)))
    elif device == "cpu":
        torch_layernorm_time = performance.CpuProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
        lib.layernorm_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_layernorm_time = \
        performance.CpuProfile((lib.layernorm_cpu, 
        (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize, byteSize)))
    elif device == "mlu":
        torch_layernorm_time = performance.BangProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
        '''
        lib.layernorm_bang.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_layernorm_time = \
        performance.BangProfile((lib.layernorm_bang, 
        (input_ptr, scale_ptr, bias_ptr, output_ptr, eps, size, behindsize, byteSize)))
        '''
        lib.layernorm_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int
        ]
        
        custom_layernorm_time = \
        performance.BangProfile((lib.layernorm_cnnl, 
        (input_ptr, scale_ptr, bias_ptr, output_ptr, ctypes_array, ndim, axis, eps, byteSize)))
    elif device == "npu":
        torch_layernorm_time = performance.AscendProfile((layer_norm.forward, (input,)))  # 以毫秒为单位
        lib.layernorm_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int
        ]
        
        custom_layernorm_time = \
        performance.BangProfile((lib.layernorm_aclnn, 
        (input_ptr, scale_ptr, bias_ptr, output_ptr, ctypes_array, ndim, axis, eps, byteSize)))
    performance.logBenchmark(torch_layernorm_time, custom_layernorm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = layer_norm.forward(input).to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test layernorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # test_shape, axis, eps
        #cpu测试用小数据
        ((7, 12, 24), 1, 1e-5),
        ((7, 12, 24), 0, 1e-5),
        ((7, 12, 24), 2, 1e-5),

        # ((700, 1200, 24), 1, 1e-5),
        # ((700, 1200, 24), 0, 1e-5),
        # ((700, 1200, 24), 2, 1e-5),
         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for test_shape, axis, eps in test_cases:
    test(test_shape, axis, eps, args.device)