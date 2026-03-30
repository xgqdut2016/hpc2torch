# ========================
# source_nvidia.cmake
# ========================

message(STATUS "Configuring NVIDIA CUDA backend")

# ------------------------
# 1. 自动获取 PyTorch 路径（通用，不写死）
# ------------------------
execute_process(
    COMMAND python -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_INCLUDE ${TORCH_DIR}/include)
set(TORCH_LIB ${TORCH_DIR}/lib)

message(STATUS "✅ 自动找到 PyTorch 路径: ${TORCH_DIR}")
message(STATUS "✅ PyTorch 头文件: ${TORCH_INCLUDE}")
message(STATUS "✅ PyTorch 库路径: ${TORCH_LIB}")

# ------------------------
# 额外添加 dlpack 头文件路径
# ------------------------
execute_process(
    COMMAND python -c "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'include'))"
    OUTPUT_VARIABLE DLPACK_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "✅ DLPack include: ${DLPACK_INCLUDE}")
include_directories(${DLPACK_INCLUDE})

add_definitions(-DENABLE_NVIDIA_API -DUSE_CUDA=1)
enable_language(CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ------------------------
# 注入 SGL_CUDA_ARCH 宏（关键）
# ------------------------
add_definitions(-DSGL_CUDA_ARCH=${SGL_CUDA_ARCH})

# CUTLASS
if(DEFINED ENV{CUTLASS_ROOT})
    include_directories($ENV{CUTLASS_ROOT})
endif()

# ------------------------
# 2. 添加 PyTorch 头文件路径
# ------------------------
include_directories(${TORCH_INCLUDE})
include_directories(${TORCH_INCLUDE}/torch/csrc/api/include)

# CUDA 源文件
file(GLOB_RECURSE NVIDIA_CUDA_SRC
    src/*/gpu/*.cu
    src/*/*/gpu/*.cu
)

file(GLOB_RECURSE NVIDIA_CPP_SRC
    src/**/gpu/*.cpp
)

set(PROJECT_CUDA_SOURCES
    ${NVIDIA_CUDA_SRC}
    ${NVIDIA_CPP_SRC}
)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


set_source_files_properties(${NVIDIA_CUDA_SRC}
    PROPERTIES
    LANGUAGE CUDA
)

# ------------------------
# 3. 链接 PyTorch 依赖库
# ------------------------
set(NVIDIA_LINK_LIBS
    ${CUDA_LIBRARIES}
    cudnn
    cublas

    # PyTorch 核心库（解决 undefined symbol）
    ${TORCH_LIB}/libtorch.so
    ${TORCH_LIB}/libtorch_cpu.so
    ${TORCH_LIB}/libtorch_cuda.so
    ${TORCH_LIB}/libc10.so
    ${TORCH_LIB}/libc10_cuda.so
    ${TORCH_LIB}/libtorch_python.so
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

message(STATUS "✅ NVIDIA 后端：已集成 PyTorch 依赖")
