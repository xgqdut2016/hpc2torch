# ========================
# source_ql.cmake （已自动集成 PyTorch 依赖）
# ========================

message(STATUS "Configuring QLCC (QL) CUDA backend")

# ------------------------
# 1. 自动获取 PyTorch 路径（关键！不写死路径）
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
# 编译宏
# ------------------------
add_compile_definitions(USE_CUDA=1 ENABLE_QL_API)

# ------------------------
# QLCC 路径 / 架构
# ------------------------
set(QLCC_PATH /usr/local/denglin/sdk/bin/dlcc)
set(QLCC_CUDA_PATH /usr/local/denglin/sdk)
set(QLCC_ARCH dlgput64)

# CUTLASS
if(DEFINED ENV{CUTLASS_ROOT})
    set(CUTLASS_INCLUDE $ENV{CUTLASS_ROOT})
endif()

# ------------------------
# 源文件
# ------------------------
file(GLOB_RECURSE QL_CUDA_SRC
    src/*/gpu/*.cu
    src/*/*/gpu/*.cu
)

file(GLOB_RECURSE QL_CPP_SRC
    src/**/gpu/*.cpp
)

# 仅 C++ 文件（包括 cudnn 调用）
set(PROJECT_QL_CPP_SOURCES ${QL_CPP_SRC})

# ------------------------
# 编译 .cu -> .o （已自动加入 PyTorch 头文件！）
# ------------------------
set(QL_CU_OBJECTS "")

foreach(cu ${QL_CUDA_SRC})
    get_filename_component(name ${cu} NAME_WE)
    get_filename_component(dir  ${cu} DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel ${dir})

    set(obj ${CMAKE_BINARY_DIR}/${rel}/${name}.cu.o)
    get_filename_component(obj_dir ${obj} DIRECTORY)

    add_custom_command(
        OUTPUT ${obj}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${obj_dir}
        COMMAND ${QLCC_PATH}
            -c -x cuda ${cu} -o ${obj}
            -DENABLE_QL_API
            --cuda-path=${QLCC_CUDA_PATH}
            --cuda-gpu-arch=${QLCC_ARCH}
            --offload-arch=${QLCC_ARCH},dlgpux64
            -I${PROJECT_SOURCE_DIR}/include
            -I${CUTLASS_INCLUDE}
            -I${QLCC_CUDA_PATH}/include
            -I${TORCH_INCLUDE}          # ✅ 自动加 PyTorch 头文件
            -I${TORCH_INCLUDE}/torch/csrc/api/include  # ✅ 关键头文件
            -O2 -std=c++17 -fPIC
            -mllvm -dlgpu-lower-xtpvn=true
        DEPENDS ${cu}
        COMMENT "QLCC compiling ${cu}"
        VERBATIM
    )

    list(APPEND QL_CU_OBJECTS ${obj})
endforeach()

# 生成目标
add_custom_target(ql_cuda_objs ALL DEPENDS ${QL_CU_OBJECTS})

# ------------------------
# 链接库（已自动加入 PyTorch 所有依赖库）
# ------------------------
set(QL_LINK_LIBS
    ${QLCC_CUDA_PATH}/lib/libcurt.so
    ${QLCC_CUDA_PATH}/lib/libcublas.so
    ${QLCC_CUDA_PATH}/lib/libcudnn.so

    # ✅ 自动链接 PyTorch 库（解决你所有 undefined symbol 问题！）
    ${TORCH_LIB}/libtorch.so
    ${TORCH_LIB}/libtorch_cpu.so
    ${TORCH_LIB}/libtorch_cuda.so
    ${TORCH_LIB}/libc10.so
    ${TORCH_LIB}/libc10_cuda.so
    ${TORCH_LIB}/libtorch_python.so
)

message(STATUS "✅ 已自动链接 PyTorch 所有库: ${QL_LINK_LIBS}")
