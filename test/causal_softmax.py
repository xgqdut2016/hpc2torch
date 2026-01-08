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
def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1, -torch.inf, y.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(type)

def test(test_shape, device):
    byteSize = 2
    if byteSize == 2:
        test_dtype = torch.float16
    elif byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Causal softmax on {device} with test_shape:{test_shape}, dtype:{test_dtype}"
    )
    ndim = len(test_shape)
    
    input = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    output = input.clone()
    
    shape_array = np.array(test_shape, dtype=np.int32)
    shape = shape_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    stride_array = np.array(list(input.stride()), dtype=np.int32)
    stride_ptr = stride_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    dimsize = test_shape[ndim - 1]
    mask = dimsize - test_shape[ndim - 2]
    othersize = 1
    for i in range(ndim - 1):
        othersize *= test_shape[i]
    
    if device == "mlu":
        torch_causal_softmax_time = performance.BangProfile((causal_softmax, (input, ))) #虽然迭代20次，但是不会修改input取值
        
        lib.causal_softmax_bang.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_causal_softmax_time = \
        performance.BangProfile((lib.causal_softmax_bang, (output_ptr, stride_ptr, shape, othersize, dimsize, mask, ndim, byteSize)))
        '''
        lib.causal_softmax_cnnl_f32.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int
        ]
        custom_causal_softmax_time = \
        performance.BangProfile((lib.causal_softmax_cnnl_f32, (output_ptr, shape, ndim)))
        '''
    elif device == "sdaa":
        torch_causal_softmax_time = performance.TecoProfile((causal_softmax, (input.to("cpu"), ))) #虽然迭代20次，但是不会修改input取值
        
        lib.causal_softmax_teco.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_causal_softmax_time = \
        performance.TecoProfile((lib.causal_softmax_teco, (output_ptr, shape, stride_ptr, ndim, mask, byteSize)))
        #print(output.flatten()[:10])
    performance.logBenchmark(torch_causal_softmax_time, custom_causal_softmax_time)
    for i in range(40):#performance里面对output迭代了40次，因此这里需要同样迭代那么多次才能是正确结果
        input = causal_softmax(input.to("cpu"))
    
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = input.to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test causal_softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'sdaa'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        ((32, 20, 128, 512)),
        ((32, 20, 512)),
        ((32, 5, 5)),
        ((32, 128)),
        ((32, 4096)),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'sdaa':
    import torch_sdaa
# 执行过滤后的测试用例
for test_shape in test_cases:
    test(test_shape, args.device)
