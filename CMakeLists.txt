cmake_minimum_required(VERSION 3.16)

project(MyCUDAProject)

# 查找 Python 库
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 添加选项控制不同的编译方式
option(USE_CUDA "Enable CUDA compilation" OFF)
option(USE_BANG "Enable BANG compilation" OFF)
option(USE_KUNLUN "Enable KUNLUN compilation" OFF)
option(USE_ASCEND "Enable ASCEND compilation" OFF)
option(USE_TECO "Enable TECO compilation" OFF)
option(USE_CPU "Enable CPU-only compilation" OFF)

set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i.86)")
    # x86架构设置
    message(STATUS "Target architecture: x86 (enabling AVX2/FMA)")
    set(ARCH_CXX_FLAGS "-mavx2 -mfma -O3")
    
    # 检查编译器是否支持AVX2
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(ARCH_CXX_FLAGS "${ARCH_CXX_FLAGS} -mavx2")
    else()
        message(WARNING "Compiler does not support AVX2 instructions")
    endif()
    
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|arm.*)")
    # ARM架构设置
    message(STATUS "Target architecture: ARM (enabling NEON)")
    set(ARCH_CXX_FLAGS "-O3 -mcpu=generic+simd")
    
    # 检查编译器是否支持NEON
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        set(ARCH_CXX_FLAGS "${ARCH_CXX_FLAGS} -mfpu=neon")
    else()
        message(WARNING "Compiler does not support NEON instructions")
    endif()
else()
    message(WARNING "Unknown architecture, using generic optimization flags")
    set(ARCH_CXX_FLAGS "-O3")
endif()

# 应用架构特定的编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_CXX_FLAGS}")
# 查找源文件
# 添加头文件搜索路径  
include_directories(${PROJECT_SOURCE_DIR}/include)

#使用 GLOB 命令找到 include/ 下的所有 .cpp 文件
file(GLOB INCLUDE_SOURCE_FILES "include/cpu/**.cpp")
file(GLOB ASCEND_INCLUDE_SOURCE_FILES "include/npu/**.cpp")#昇腾机器头文件

file(GLOB KUNLUN_INCLUDE_SOURCE_FILES "include/kunlun/**.xpu")
 
# 使用 list(APPEND ...) 命令将 INCLUDE_SOURCE_FILES 添加到 CPP_SOURCE_FILES
file(GLOB CPP_SOURCE_FILES "src/**/cpu/*.cpp")
list(APPEND CPP_SOURCE_FILES ${INCLUDE_SOURCE_FILES})

file(GLOB ASCEND_SOURCE_FILES "src/**/npu/*.cpp")
list(APPEND ASCEND_SOURCE_FILES ${INCLUDE_SOURCE_FILES} ${ASCEND_INCLUDE_SOURCE_FILES})

file(GLOB KUNLUN_SOURCE_FILES "src/**/kunlun/*.xpu")
list(APPEND KUNLUN_SOURCE_FILES ${KUNLUN_INCLUDE_SOURCE_FILES})

file(GLOB GPU_CUDA_FILES "src/**/gpu/*.cu")
file(GLOB GPU_CUDNN_FILES "src/**/gpu/*.cpp")
list(APPEND CUDA_SOURCE_FILES ${GPU_CUDA_FILES} ${GPU_CUDNN_FILES})

# 查找所有 .mlu, .cpp 文件 
file(GLOB BANG_MLU_FILES "src/**/mlu/*.mlu")
file(GLOB BANG_CNNL_FILES "src/**/mlu/*.cpp")
list(APPEND BANG_SOURCE_FILES ${BANG_MLU_FILES} ${BANG_CNNL_FILES})

file(GLOB TECO_SCPP_FILES "src/**/teco/*.scpp")
list(APPEND TECO_SOURCE_FILES ${TECO_SCPP_FILES})

# 根据选项决定编译哪些源文件
if(USE_CUDA)
    # 设置 CUDA 编译选项
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    message(STATUS "CUDA build enabled.")
    enable_language(CXX)
    enable_language(CUDA)
    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})#寻找cuda以及cudnn等安装路径
    set_target_properties(my_library PROPERTIES
        CUDA_ARCHITECTURES 80
    )
    target_link_libraries(my_library ${CUDA_LIBRARIES} cudnn cublas)#如果调用cudnn库需要链接cudnn, cublas

elseif(USE_BANG)
    message(STATUS "BANG build enabled.")
    
    set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++17 -pthread -pipe")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")

    # check `NEUWARE_HOME` env
    if(NOT DEFINED ENV{NEUWARE_HOME})  
        set(NEUWARE_HOME "/usr/local/neuware" CACHE PATH "Path to NEUWARE installation")  
    else()  
        set(NEUWARE_HOME $ENV{NEUWARE_HOME} CACHE PATH "Path to NEUWARE installation" FORCE)  
    endif()
      # check `NEUWARE_HOME` env
    message(${NEUWARE_HOME})
    if(EXISTS ${NEUWARE_HOME})
        include_directories("${NEUWARE_HOME}/include")
        link_directories("${NEUWARE_HOME}/lib64")
        link_directories("${NEUWARE_HOME}/lib")
        set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
    else()
        message(FATAL_ERROR "NEUWARE directory cannot be found, refer README.md to prepare NEUWARE_HOME environment.")
    endif()

    # setup cmake search path
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake"
    "${NEUWARE_HOME}/cmake"
    "${NEUWARE_HOME}/cmake/modules"
    )

    # include FindBANG.cmake and check cncc
    find_package(BANG)
    if(NOT BANG_FOUND)
        message(FATAL_ERROR "BANG cannot be found.")
    elseif (NOT BANG_CNCC_EXECUTABLE)
        message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should check path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
    endif()

    # setup cncc flags
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -Wall -Werror -std=c++17 -pthread")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_592")

    list(APPEND ALL_SOURCE_FILES ${BANG_SOURCE_FILES} ${CPP_SOURCE_FILES})
    bang_add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
    target_link_libraries(my_library cnnl cnnl_extra cnrt cndrv)
elseif(USE_ASCEND)
    message(STATUS "ASCEND build enabled.")
    add_compile_definitions(USE_ASCEND=1)
    if ((NOT DEFINED ASCEND_HOME) AND (NOT DEFINED ENV{ASCEND_HOME}))
        message(FATAL_ERROR "ASCEND_HOME is not defined from cmake or env")
    elseif (DEFINED ASCEND_HOME)
        set(ASCEND_HOME ${ASCEND_HOME} CACHE STRING "ASCEND_HOME directory for Ascend development")
    else()
        set(ASCEND_HOME $ENV{ASCEND_HOME} CACHE STRING "ASCEND_HOME directory for Ascend development")
    endif()
    message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

    include_directories("${ASCEND_HOME}/include/")
    include_directories("${ASCEND_HOME}/include/aclnn")
    find_library(ASCEND_CL libascendcl.so "${ASCEND_HOME}/lib64")
    find_library(ASCEND_BASE libnnopbase.so "${ASCEND_HOME}/lib64")
    find_library(ASCEND_DNN libopapi.so "${ASCEND_HOME}/lib64")
    find_library(ASCEND_HCCL libhccl.so "${ASCEND_HOME}/lib64")
    find_library(ASCEND_HAL libascend_hal.so "${ASCEND_HOME}/../../driver/lib64/driver")
    # find_library(ASCEND_RT libruntime.so "${ASCEND_HOME}/lib64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++ -Wall -Werror -fopenmp")
    if ((NOT DEFINED TARGET_CPU_ARCH) AND (NOT DEFINED ENV{TARGET_CPU_ARCH}))
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _uname_m OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(TARGET_CPU_ARCH "${_uname_m}" CACHE STRING "Target CPU ARCH")
    elseif(DEFINED TARGET_CPU_ARCH)
        set(TARGET_CPU_ARCH ${TARGET_CPU_ARCH} CACHE STRING "Target CPU ARCH")
    else()
        set(TARGET_CPU_ARCH $ENV{TARGET_CPU_ARCH} CACHE STRING "Target CPU ARCH")
    endif()
    message(STATUS "TARGET_CPU_ARCH: ${TARGET_CPU_ARCH}")
    list(APPEND ALL_SOURCE_FILES ${ASCEND_SOURCE_FILES} ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
    target_link_libraries(my_library ${ASCEND_HAL} ${ASCEND_CL} ${ASCEND_BASE} ${ASCEND_DNN} ${ASCEND_HCCL} stdc++)

# 设置 CMake C++ 编译器为 Kunlun 提供的 clang++ 编译器
elseif(USE_KUNLUN)
    # set(CMAKE_C_COMPILER "/usr/local/xpu/XTDK/bin/clang")
    # set(CMAKE_CXX_COMPILER "/usr/local/xpu/XTDK/bin/clang++")

    message(STATUS "Kunlun build enabled.")

    set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++17 -pthread -pipe")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")

    # 设置 Kunlun SDK 路径
    if(NOT DEFINED ENV{KUNLUN_HOME})
        set(KUNLUN_HOME "/usr/local/kunlun" CACHE PATH "Path to Kunlun SDK")
    else()
        set(KUNLUN_HOME $ENV{KUNLUN_HOME} CACHE PATH "Path to Kunlun SDK" FORCE)
    endif()

    message(STATUS "KUNLUN_HOME=${KUNLUN_HOME}")
    if(EXISTS ${KUNLUN_HOME})
        include_directories("${KUNLUN_HOME}/include")
        link_directories("${KUNLUN_HOME}/lib64")
    else()
        message(FATAL_ERROR "Kunlun SDK not found. Please set KUNLUN_HOME environment variable.")
    endif()

    set(XTDK_DIR "${KUNLUN_HOME}/XTDK")

    # 设置编译 .xpu 文件的规则
    set(KUNLUN_XPU_OBJECTS "")

    # 设置中间文件输出目录
    set(KUNLUN_OBJ_DIR "${CMAKE_BINARY_DIR}/kunlun_objs")

    file(MAKE_DIRECTORY ${KUNLUN_OBJ_DIR})  # 创建目录

    foreach(xpu_file ${KUNLUN_SOURCE_FILES})
        get_filename_component(fname ${xpu_file} NAME_WE)

        # 把中间产物移到 kunlun_objs 目录
        set(obj_base "${KUNLUN_OBJ_DIR}/${fname}")
        set(object_file "${obj_base}.o")
        set(bin_object "${obj_base}.device.bin.o")
        
        add_custom_command(
            OUTPUT ${object_file} ${bin_object}
            COMMAND ${XTDK_DIR}/bin/clang++
                --sysroot=/ --target=aarch64-linux-gnu
                -fPIC -pie --xpu-arch=xpu2
                --basename ${obj_base} -std=c++11 -O2 -fno-builtin -g -c ${xpu_file} -v
                -I${CMAKE_SOURCE_DIR}/include
                -I${KUNLUN_HOME}/include
            DEPENDS ${xpu_file}
            COMMENT "Compiling Kunlun XPU kernel ${xpu_file}"
        )

        list(APPEND KUNLUN_XPU_OBJECTS ${object_file} ${bin_object})
    endforeach()

    # 定义目标
    add_custom_target(kunlun_xpu_kernels ALL DEPENDS ${KUNLUN_XPU_OBJECTS})

    # 编译其他源代码并链接库
    add_library(my_library SHARED ${KUNLUN_SOURCE_FILES} ${KUNLUN_XPU_OBJECTS})
    add_dependencies(my_library kunlun_xpu_kernels)
    target_compile_options(my_library PRIVATE -Wall -Werror -fPIC)
    target_include_directories(my_library PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(my_library PRIVATE stdc++ xpuapi xpurt)


    # 设置 C++ 标准
    set_target_properties(my_library PROPERTIES CXX_STANDARD 17)
    set_target_properties(my_library PROPERTIES LINKER_LANGUAGE CXX)
    
elseif(USE_TECO)
    # set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "TECO build enabled.")
    add_compile_definitions(ENABLE_TECO_SDAA)

    if ((NOT DEFINED TECO_HOME) AND (NOT DEFINED ENV{TECO_HOME}))
        set(TECO_HOME "/opt/tecoai")
        message(WARNING "TECO_HOME not defined, defaulting to /opt/tecoai")
    elseif (DEFINED TECO_HOME)
        set(TECO_HOME ${TECO_HOME} CACHE STRING "TECO SDK directory")
    else()
        set(TECO_HOME $ENV{TECO_HOME} CACHE STRING "TECO SDK directory")
    endif()
    message(STATUS "TECO_HOME: ${TECO_HOME}")

    include_directories(${TECO_HOME}/include)
    link_directories(${TECO_HOME}/lib64)

    # 使用已有的 TECO_SOURCE_FILES 列表，编译 .scpp 文件
    set(SCPP_OBJECTS "")
    foreach(scpp_file ${TECO_SOURCE_FILES})
        get_filename_component(fname ${scpp_file} NAME_WE)
        set(obj_file ${CMAKE_CURRENT_BINARY_DIR}/${fname}.o)
        add_custom_command(
            OUTPUT ${obj_file}
            COMMAND ${TECO_HOME}/bin/tecocc ${scpp_file} -o ${obj_file}
                -O2 -fPIC -Wall -Werror -std=c++17 -pthread -c
                -I${TECO_HOME}/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
            DEPENDS ${scpp_file}
            COMMENT "Compiling ${scpp_file} with tecocc"
        )
        list(APPEND SCPP_OBJECTS ${obj_file})
    endforeach()

    # 创建 teco_objects 构建目标
    add_custom_target(teco_objects ALL DEPENDS ${SCPP_OBJECTS})

    # 添加最终目标
    list(APPEND ALL_SOURCE_FILES ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES} ${SCPP_OBJECTS})
    add_dependencies(my_library teco_objects)
    # target_compile_options(my_library PRIVATE -fsanitize=address -g3 -O0)
    # target_link_options(my_library PRIVATE -fsanitize=address)
    # 设置编译标准和选项
    find_package(OpenMP REQUIRED)
    target_compile_features(my_library PRIVATE cxx_std_17)
    target_compile_options(my_library PRIVATE -Wall -Werror -fPIC)
    target_include_directories(my_library PRIVATE ${TECO_HOME}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # 链接 TECO SDK 库
    target_link_libraries(my_library
        OpenMP::OpenMP_CXX
        stdc++
        ${TECO_HOME}/lib64/libsdaart.so
        ${TECO_HOME}/lib64/libtecoblas.so
        ${TECO_HOME}/lib64/libtecodnn.so
    )

elseif(USE_CPU)
    message(STATUS "CPU-only build enabled.")
    enable_language(CXX)
    list(APPEND ALL_SOURCE_FILES ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
else()
    message(FATAL_ERROR "No valid compilation mode specified. Please enable USE_CUDA, USE_BANG, or USE_CPU.")
endif()




# 设置编译选项
target_compile_features(my_library PUBLIC cxx_std_11)

# 链接 Python 库
target_link_libraries(my_library PRIVATE ${Python3_LIBRARIES})

# 指定输出目录
set_target_properties(my_library PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
