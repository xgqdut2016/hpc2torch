# ========================
# source_ascend.cmake
# ========================
message(STATUS "Configuring ASCEND backend")

add_compile_definitions(USE_ASCEND=1)

# ------------------------
# ASCEND_HOME
# ------------------------
if((NOT DEFINED ASCEND_HOME) AND (NOT DEFINED ENV{ASCEND_HOME}))
    message(FATAL_ERROR "ASCEND_HOME is not defined from cmake or environment")
elseif(DEFINED ASCEND_HOME)
    set(ASCEND_HOME ${ASCEND_HOME} CACHE STRING "ASCEND_HOME directory for Ascend development")
else()
    set(ASCEND_HOME $ENV{ASCEND_HOME} CACHE STRING "ASCEND_HOME directory for Ascend development")
endif()

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

# ------------------------
# Include directories
# ------------------------
include_directories(
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/aclnn
    ${CMAKE_SOURCE_DIR}/include
)

# ------------------------
# Find Ascend libraries
# ------------------------
find_library(ASCEND_CL libascendcl.so "${ASCEND_HOME}/lib64")
find_library(ASCEND_BASE libnnopbase.so "${ASCEND_HOME}/lib64")
find_library(ASCEND_DNN libopapi.so "${ASCEND_HOME}/lib64")
find_library(ASCEND_HCCL libhccl.so "${ASCEND_HOME}/lib64")
find_library(ASCEND_HAL libascend_hal.so "${ASCEND_HOME}/../../driver/lib64/driver")

if(NOT ASCEND_CL OR NOT ASCEND_BASE OR NOT ASCEND_DNN OR NOT ASCEND_HCCL OR NOT ASCEND_HAL)
    message(FATAL_ERROR "Cannot find required Ascend libraries in ${ASCEND_HOME}/lib64")
endif()

set(ASCEND_LINK_LIBS
    ${ASCEND_HAL} ${ASCEND_CL} ${ASCEND_BASE} ${ASCEND_DNN} ${ASCEND_HCCL} stdc++
)

# ------------------------
# Source files
# ------------------------
file(GLOB ASCEND_INCLUDE_SOURCE_FILES "${CMAKE_SOURCE_DIR}/include/npu/**/*.cpp")
file(GLOB ASCEND_SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/**/npu/*.cpp")
list(APPEND ASCEND_SOURCE_FILES ${CPP_SOURCE_FILES} ${ASCEND_INCLUDE_SOURCE_FILES})

set(PROJECT_ASCEND_SOURCES ${ASCEND_SOURCE_FILES})

# ------------------------
# Compiler flags
# ------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fPIC -std=c++17 -fopenmp")

