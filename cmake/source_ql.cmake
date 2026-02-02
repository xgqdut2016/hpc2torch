# ========================
# source_ql.cmake
# ========================

message(STATUS "Configuring DLCC (QL) CUDA backend")

# ------------------------
# 编译宏
# ------------------------
add_compile_definitions(USE_CUDA=1 ENABLE_QL_API)

# ------------------------
# DLCC 路径 / 架构
# ------------------------
set(DLCC_PATH /usr/local/denglin/sdk/bin/dlcc)
set(DLCC_CUDA_PATH /usr/local/denglin/sdk)
set(DLCC_ARCH dlgput64)

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
# 编译 .cu -> .o
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
        COMMAND ${DLCC_PATH}
            -c -x cuda ${cu} -o ${obj}
            -DENABLE_QL_API
            --cuda-path=${DLCC_CUDA_PATH}
            --cuda-gpu-arch=${DLCC_ARCH}
            --offload-arch=${DLCC_ARCH},dlgpux64
            -I${PROJECT_SOURCE_DIR}/include
            -I${CUTLASS_INCLUDE}
            -I${DLCC_CUDA_PATH}/include   # ✅ 这里加上 cudnn.h 所在 include
            -O2 -std=c++17 -fPIC
            -mllvm -dlgpu-lower-xtpvn=true
        DEPENDS ${cu}
        COMMENT "DLCC compiling ${cu}"
        VERBATIM
    )

    list(APPEND QL_CU_OBJECTS ${obj})
endforeach()

# 生成目标
add_custom_target(ql_cuda_objs ALL DEPENDS ${QL_CU_OBJECTS})

# ------------------------
# 链接库
# ------------------------
set(QL_LINK_LIBS
    ${DLCC_CUDA_PATH}/lib/libcurt.so
    ${DLCC_CUDA_PATH}/lib/libcublas.so
    ${DLCC_CUDA_PATH}/lib/libcudnn.so
)
