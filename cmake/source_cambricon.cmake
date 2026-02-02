# ========================
# source_cambricon.cmake
# ========================

message(STATUS "Configuring Cambricon (BANG) backend")

# 添加宏定义
add_compile_definitions(USE_BANG=1)

# ------------------------
# NEUWARE_HOME
# ------------------------
if(NOT DEFINED ENV{NEUWARE_HOME})
    set(NEUWARE_HOME "/usr/local/neuware" CACHE PATH "Path to NEUWARE installation")
else()
    set(NEUWARE_HOME $ENV{NEUWARE_HOME} CACHE PATH "Path to NEUWARE installation" FORCE)
endif()

message(STATUS "NEUWARE_HOME = ${NEUWARE_HOME}")

if(EXISTS ${NEUWARE_HOME})
    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib64")
    link_directories("${NEUWARE_HOME}/lib")
    set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
else()
    message(FATAL_ERROR "NEUWARE directory cannot be found. Please set NEUWARE_HOME correctly.")
endif()

# ------------------------
# CMake module path
# ------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake"
    "${NEUWARE_HOME}/cmake"
    "${NEUWARE_HOME}/cmake/modules"
)

# ------------------------
# Find BANG / cncc
# ------------------------
find_package(BANG)
if(NOT BANG_FOUND)
    message(FATAL_ERROR "BANG cannot be found.")
elseif(NOT BANG_CNCC_EXECUTABLE)
    message(FATAL_ERROR "cncc not found. Ensure BANG_CNCC_EXECUTABLE is set or in PATH.")
endif()

# ------------------------
# CNCC flags
# ------------------------
set(BANG_CNCC_FLAGS "-fPIC -Wall -Werror -std=c++17 -pthread -O3 --bang-mlu-arch=mtp_592")

# ------------------------
# Source files
# ------------------------
file(GLOB_RECURSE BANG_SOURCE_FILES
    src/**/mlu/*.mlu
    src/**/mlu/*.cpp
    include/**/mlu/*.xpu
)

set(PROJECT_BANG_SOURCES ${BANG_SOURCE_FILES})

# ------------------------
# Library helper
# ------------------------
# bang_add_library 是寒武纪官方 CMake helper
# 可以在 CMakeLists.txt 里直接调用
set(BANG_LINK_LIBS cnnl cnnl_extra cnrt cndrv)

