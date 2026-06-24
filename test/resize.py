import torch
import torchvision
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

def crop_and_resize(input_image, boxes, box_indices, output_size, mode='bilinear'):
    cropped_images = []
    
    for box, index in zip(boxes, box_indices):
        top, left, bottom, right = box
        cropped_image = torchvision.transforms.functional.crop(input_image[index], top, left, bottom - top, right - left)
        resized_image = torch.nn.functional(cropped_image.unsqueeze(0), size=output_size, mode=mode, align_corners=False)
        cropped_images.append(resized_image)

    return torch.cat(cropped_images, dim=0)

def test(inputShape, roi, device):
    operator = "nearest"
    byteSize = 2
    
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing {operator} reduce on {device} with inputShape:{inputShape}, roi:{roi}, dtype:{tensor_dtype}"
    )
    
    
    a = torch.rand(inputShape, dtype=tensor_dtype).to(device)
    ndim = len(inputShape)
    
    aData = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    aShape = np.array(inputShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    
    if operator == "nearest":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((maxReduce, (a, axes)))  
            lib.nearest_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.nearest_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = maxReduce(a, axes).to('cpu').numpy().flatten()
