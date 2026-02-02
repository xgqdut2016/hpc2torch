# ========================
# common.cmake
# ========================

set(CMAKE_CXX_STANDARD 17)

# Python
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 架构相关优化
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i.86)")
    message(STATUS "Target architecture: x86")
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    if(HAS_AVX2)
        add_compile_options(-O3 -mavx2 -mfma)
    else()
        add_compile_options(-O3)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|arm.*)")
    message(STATUS "Target architecture: ARM")
    add_compile_options(-O3 -mcpu=generic+simd)
else()
    add_compile_options(-O3)
endif()

# include 目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# ------------------------
# CPU 源文件
# ------------------------
file(GLOB_RECURSE CPU_SRC
    src/**/cpu/*.cpp
    include/cpu/*.cpp
)

set(PROJECT_CPU_SOURCES ${CPU_SRC})

