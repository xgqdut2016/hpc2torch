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
# 从环境变量获取 TVM 路径（对齐 xmake.lua 逻辑：TVM_ROOT / TVM_HOME / TVM_PATH）
# ------------------------
set(TVM_ROOT "$ENV{TVM_ROOT}")
if(NOT TVM_ROOT)
    set(TVM_ROOT "$ENV{TVM_HOME}")
endif()
if(NOT TVM_ROOT)
    set(TVM_ROOT "$ENV{TVM_PATH}")
endif()

# 如果找到 TVM_ROOT，添加头文件 + 定义宏
if(TVM_ROOT)
    message(STATUS "✅ 找到 TVM 根目录: ${TVM_ROOT}")
    
    # 定义宏 ENABLE_TVM_API（对齐 xmake）
    add_definitions(-DENABLE_TVM_API)
    
    # 添加头文件搜索路径（完全对齐 xmake 的 3 个路径）
    include_directories(${TVM_ROOT})
    include_directories(${TVM_ROOT}/include)
    include_directories(${TVM_ROOT}/3rdparty/dlpack/include)
    
    message(STATUS "✅ 已添加 TVM 头文件路径")
    message(STATUS "✅ 已启用宏: ENABLE_TVM_API")
    # 计算 SGL_CUDA_ARCH = major*100 + minor*10
    math(EXPR SGL_CUDA_ARCH "${CUDA_ARCH} * 10")

    message(STATUS "✅ SGL_CUDA_ARCH = ${SGL_CUDA_ARCH}")
else()
    message(STATUS "ℹ️ 未设置 TVM_ROOT / TVM_HOME / TVM_PATH，不启用 TVM 相关功能")
endif()

message(STATUS "✅ DLPack include: ${DLPACK_INCLUDE}")
include_directories(${DLPACK_INCLUDE})

add_definitions(-DENABLE_NVIDIA_API -DUSE_CUDA=1)
enable_language(CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --threads 8")

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

message(STATUS "✅ NVIDIA 后端：已集成 PyTorch 依赖")

# ========================
# 自动生成 AWQ Marlin Kernels (仅首次生成，已存在则跳过)
# ========================
set(GENERATE_SCRIPT ${PROJECT_SOURCE_DIR}/src/awq_marlin_gemm/gpu/generate_kernels.py)
set(GENERATED_HEADER ${PROJECT_SOURCE_DIR}/src/awq_marlin_gemm/gpu/kernel_selector.h)

# 如果头文件已存在，则跳过生成
if(EXISTS ${GENERATED_HEADER})
    message(STATUS "✅ AWQ Marlin 内核已生成，跳过重复生成 (${GENERATED_HEADER})")
else()
    # 将数字架构转为浮点数 80 -> 8.0, 90 ->9.0
    set(GENERATE_ARCH "${CUDA_ARCH}.0")
    
    message(STATUS "🔧 首次生成 AWQ Marlin 内核 (架构: ${GENERATE_ARCH})")
    
    # 执行 Python 生成内核头文件
    execute_process(
        COMMAND python ${GENERATE_SCRIPT} ${GENERATE_ARCH}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE GEN_KERNEL_RESULT
        OUTPUT_VARIABLE GEN_KERNEL_OUTPUT
        ERROR_VARIABLE GEN_KERNEL_ERROR
    )
    
    # 检查是否生成成功
    if(NOT GEN_KERNEL_RESULT EQUAL 0)
        message(FATAL_ERROR "❌ 生成 AWQ Marlin 内核失败！\n错误信息: ${GEN_KERNEL_ERROR}")
    endif()
    
    message(STATUS "✅ 首次生成 AWQ Marlin 内核完成！")
endif()
