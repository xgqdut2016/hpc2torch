cmake_minimum_required(VERSION 3.16)

project(MyCUDAProject)

# 查找 Python 库
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 设置 CUDA 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# 添加选项控制不同的编译方式
option(USE_CUDA "Enable CUDA compilation" OFF)
option(USE_BANG "Enable BANG compilation" OFF)
option(USE_CPU "Enable CPU-only compilation" OFF)

# 查找源文件
# 添加头文件搜索路径  
include_directories(${PROJECT_SOURCE_DIR}/include)

#使用 GLOB 命令找到 include/ 下的所有 .cpp 文件
file(GLOB INCLUDE_SOURCE_FILES "include/**.cpp")
 
# 使用 list(APPEND ...) 命令将 INCLUDE_SOURCE_FILES 添加到 CPP_SOURCE_FILES
file(GLOB CPP_SOURCE_FILES "src/**/cpu/*.cpp")
list(APPEND CPP_SOURCE_FILES ${INCLUDE_SOURCE_FILES})


file(GLOB GPU_CUDA_FILES "src/**/gpu/*.cu")
file(GLOB GPU_CUDNN_FILES "src/**/gpu/*.cpp")
list(APPEND CUDA_SOURCE_FILES ${GPU_CUDA_FILES} ${GPU_CUDNN_FILES})

# 查找所有 .mlu, .cpp 文件 
file(GLOB BANG_MLU_FILES "src/**/mlu/*.mlu")
file(GLOB BANG_CNNL_FILES "src/**/mlu/*.cpp")
list(APPEND BANG_SOURCE_FILES ${BANG_MLU_FILES} ${BANG_CNNL_FILES})

# 根据选项决定编译哪些源文件
if(USE_CUDA)
    message(STATUS "CUDA build enabled.")
    enable_language(CXX)
    enable_language(CUDA)
    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
    add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
    target_link_libraries(my_library cudnn)#如果调用cudnn库需要链接cudnn

elseif(USE_BANG)
    message(STATUS "BANG build enabled.")
    
    set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++11 -pthread -pipe")
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
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -Wall -Werror -std=c++11 -pthread")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_592")

    list(APPEND ALL_SOURCE_FILES ${BANG_SOURCE_FILES} ${CPP_SOURCE_FILES})
    bang_add_library(my_library SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
    target_link_libraries(my_library cnnl cnnl_extra cnrt cndrv)
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
