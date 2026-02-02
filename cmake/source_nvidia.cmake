# ========================
# source_nvidia.cmake
# ========================

message(STATUS "Configuring NVIDIA CUDA backend")

add_definitions(-DENABLE_NVIDIA_API)
enable_language(CUDA)

# CUTLASS
if(DEFINED ENV{CUTLASS_ROOT})
    include_directories($ENV{CUTLASS_ROOT})
endif()

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

set(CUDA_ARCH 80)

set_source_files_properties(${NVIDIA_CUDA_SRC}
    PROPERTIES
    LANGUAGE CUDA
)

set(NVIDIA_LINK_LIBS
    ${CUDA_LIBRARIES}
    cudnn
    cublas
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


