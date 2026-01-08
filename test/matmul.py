import torch
import ctypes
import torch.nn.functional as F
import argparse
import numpy as np
from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os



lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def matmul(_c, beta, _a, _b, alpha):
    a = _a.clone()
    b = _b.clone()
    c = _c.clone()
    input_dtype = c.dtype
    ans = (
        alpha * torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(input_dtype)
        + beta * c
    )
    return ans
    
def test_diaoku(a_shape, b_shape, c_shape, alpha, beta, device):
    byteSize = 4
    if (byteSize == 4):
        test_dtype = torch.float32
    
    elif (byteSize == 2):
        test_dtype = torch.float16
    
    print(
        f"Testing matmul on {device} with alpha:{alpha}, beta:{beta}, a_shape:{a_shape} b_shape:{b_shape} c_shape:{c_shape} , dtype:{test_dtype}"
    )
    A = torch.randn(a_shape, device=device, dtype=test_dtype, requires_grad=False) 
    B = torch.randn(b_shape, device=device, dtype=test_dtype, requires_grad=False)
    C = torch.randn(c_shape, device=device, dtype=test_dtype, requires_grad=False)
    C_clone = C.clone()
    

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    aShape = np.array(a_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bShape = np.array(b_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(c_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    aDim = len(a_shape)
    bDim = len(b_shape)
    cDim = len(c_shape)

    if device == "mlu":
        torch_matmul_time = performance.BangProfile((matmul, (C_clone, beta, A, B, alpha))) 
        lib.matmul_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]           
        custom_matmul_time = \
        performance.BangProfile((lib.matmul_cnnl, 
        (A_ptr, B_ptr, C_ptr, aShape, bShape, cShape, aDim, bDim, cDim, alpha, beta, byteSize)))
        
    elif device == "npu":
        torch_matmul_time = performance.AscendProfile((matmul, (C_clone, beta, A, B, alpha))) 
        lib.matmul_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]           
        custom_matmul_time = \
        performance.AscendProfile((lib.matmul_aclnn, 
        (A_ptr, B_ptr, C_ptr, aShape, bShape, cShape, aDim, bDim, cDim, alpha, beta, byteSize)))
    elif device == "cuda":
        torch_matmul_time = performance.CudaProfile((matmul, (C_clone, beta, A, B, alpha))) 
        lib.matmul_cudnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]           
        custom_matmul_time = \
        performance.CudaProfile((lib.matmul_cudnn, 
        (A_ptr, B_ptr, C_ptr, aShape, bShape, cShape, aDim, bDim, cDim, alpha, beta, byteSize)))
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)

    for i in range(40): #对于alpha , beta > 0的情况，此时需要特别注意
        C_clone = matmul(C_clone, beta, A, B, alpha)
    tmpa = C_clone.to('cpu').detach().numpy().flatten()
    
    tmpb = C.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
def test_cuda(M, K, N, test_dtype):
    device = "cuda"
    print(
        f"Testing Attention on {device} with M-K-N:{M, K, N} , dtype:{test_dtype}"
    )
    A = torch.randn([M, K], device=device, dtype=torch.float32, requires_grad=False) 
    B = torch.randn([K, N], device=device, dtype=torch.float32, requires_grad=False)
    C = torch.zeros([M, N], device=device, dtype=torch.float32, requires_grad=False)
    

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if device == "cuda":
        torch_matmul_time = performance.CudaProfile((torch.matmul, (A, B)))
        
        lib.matmul_cuda_f32.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]
        custom_matmul_time = performance.CudaProfile((
            lib.matmul_cuda_f32,
            (A_ptr, B_ptr, C_ptr, M, K, N)
        ))
        
        
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)
    tmpa = torch.matmul(A, B).to('cpu').numpy().flatten()
    tmpb = C.to('cpu').numpy().flatten()
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test matmul on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()   

# if args.device == "cuda":
#     test_cases = [
#         # M, K, N, test_dtype, device
#         (1024, 128, 1024, torch.float32),
#         (1024, 256, 1024, torch.float32),
#         (1024, 512, 1024, torch.float32),
#         (1024, 1024, 1024, torch.float32),
#     ] 
#     for M, K, N, test_dtype in test_cases:
#         test_cuda(M, K, N, test_dtype)

#昇腾机器对高维matmul不支持alpha, beta参数
test_cases = [
    # alpha, beta, a_shape, b_shape, c_shape
    # (1.0, 0.0, (6, 2048), (2048, 2048), (6, 2048)),
    # (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048)),
    # (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048)),
    # (1.0, 0.0, (6, 2048), (2048, 2560), (6, 2560)),
    # (1.0 , 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6)),
    (1.0, 0.0, (12, 512, 1024), (1024, 1024), (12, 512, 1024)),
    (1.0, 0.0, (1, 512, 1024), (1024, 1024), (1, 512, 1024)),
]
if args.device == "mlu":
    import torch_mlu
elif args.device == "npu":
    import torch_npu
for alpha, beta, a_shape, b_shape, c_shape in test_cases:
    test_diaoku(a_shape, b_shape, c_shape, alpha, beta, args.device)
