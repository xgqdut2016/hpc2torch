import torch
import ctypes
import numpy as np
import torch.nn.functional as F
import argparse

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

# 定义函数参数类型
def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)


def test(test_shape, test_dtype, device):
    print(
        f"Testing Attention on {device} with x_shape:{test_shape} , dtype:{test_dtype}"
    )
    N, d = test_shape[0], test_shape[1]
    Q = torch.randn(test_shape, device=device, dtype=torch.float32, requires_grad=False) 
    K = torch.randn(test_shape, device=device, dtype=torch.float32, requires_grad=False)
    V = torch.randn(test_shape, device=device, dtype=torch.float32, requires_grad=False)
    # 创建输出张量
    attHPC = torch.zeros(test_shape, device = device, dtype = torch.float32)

    Q_ptr = ctypes.cast(Q.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    K_ptr = ctypes.cast(K.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    V_ptr = ctypes.cast(V.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    attHPC_ptr = ctypes.cast(attHPC.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if device == "cuda":
        lib.attention_nv_f32.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p)
        ]

        torch_flash_time = performance.CudaProfile((funAttention, (Q, K, V)))
        # 调用 C 函数
        custom_attention_time = performance.CudaProfile((
            lib.attention_nv_f32,
            (Q_ptr, K_ptr, V_ptr, N, d, attHPC_ptr)
        ))
        performance.logBenchmark(torch_flash_time, custom_attention_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = funAttention(Q, K, V).to('cpu').numpy().flatten()
    tmpb = attHPC.to('cpu').numpy().flatten()
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # x_shape, axis
        ((128, 128), torch.float32, 'cuda'),
        ((256, 128), torch.float32, 'cuda'), 
        ((1024, 128), torch.float32, 'cuda'), 
        ((1024, 1024), torch.float32, 'cuda'), 
]
filtered_test_cases = [
    (test_shape, test_dtype, device)
    for test_shape, test_dtype, device in test_cases
    if device == args.device
]

for test_shape,test_dtype, device in filtered_test_cases:
    test(test_shape, test_dtype, device)
