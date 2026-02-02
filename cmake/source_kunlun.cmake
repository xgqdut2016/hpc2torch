# ========================
# source_kunlun.cmake
# ========================

message(STATUS "Configuring Kunlun / XPU backend")

# ------------------------
# 环境
# ------------------------
if(NOT DEFINED ENV{KUNLUN_HOME})
    set(KUNLUN_HOME "/usr/local/kunlun" CACHE PATH "Path to Kunlun SDK")
else()
    set(KUNLUN_HOME $ENV{KUNLUN_HOME} CACHE PATH "Path to Kunlun SDK" FORCE)
endif()

message(STATUS "KUNLUN_HOME=${KUNLUN_HOME}")

if(NOT EXISTS ${KUNLUN_HOME})
    message(FATAL_ERROR "Kunlun SDK not found. Please set KUNLUN_HOME environment variable.")
endif()

set(XTDK_DIR "${KUNLUN_HOME}/xtdk")
set(XRE_DIR "${KUNLUN_HOME}/xre")
set(XDNN_DIR "${KUNLUN_HOME}/xhpc/xdnn")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${KUNLUN_HOME}/include
    ${XRE_DIR}/include
    ${XDNN_DIR}/include
    ${XTDK_DIR}/include
)

link_directories(
    ${XRE_DIR}/so
    ${XDNN_DIR}/so
    ${KUNLUN_HOME}/lib64
)

# ------------------------
# 搜集源文件
# ------------------------
file(GLOB_RECURSE KUNLUN_CPP_SOURCE_FILES "src/**/kunlun/*.cpp")
file(GLOB_RECURSE KUNLUN_INCLUDE_SOURCE_FILES "include/kunlun/**.xpu")
file(GLOB_RECURSE KUNLUN_SRC_SOURCE_FILES "src/**/kunlun/*.xpu")
list(APPEND KUNLUN_XPU_SOURCE_FILES ${KUNLUN_INCLUDE_SOURCE_FILES} ${KUNLUN_SRC_SOURCE_FILES})

# 中间对象目录
set(KUNLUN_OBJ_DIR "${CMAKE_BINARY_DIR}/kunlun_objs")
file(MAKE_DIRECTORY ${KUNLUN_OBJ_DIR})

set(KUNLUN_XPU_OBJECT_FILES "")
set(KUNLUN_XPU_BIN_OBJECT_FILES "")

foreach(xpu_file ${KUNLUN_XPU_SOURCE_FILES})
    get_filename_component(fname ${xpu_file} NAME_WE)
    set(obj_file "${KUNLUN_OBJ_DIR}/${fname}.o")
    set(bin_file "${KUNLUN_OBJ_DIR}/${fname}.device.bin.o")

    add_custom_command(
        OUTPUT ${obj_file} ${bin_file}
        COMMAND ${XTDK_DIR}/bin/clang++
                -fPIC
                --xpu-arch=xpu3
                -x xpu
                --basename ${KUNLUN_OBJ_DIR}/${fname}
                -std=c++17 -O2 -fno-builtin -g
                -c ${xpu_file}
                -I${CMAKE_SOURCE_DIR}/include
                -I${KUNLUN_HOME}/include
                -I${XRE_DIR}/include
                -I${XDNN_DIR}/include
                -I${XTDK_DIR}/include
        DEPENDS ${xpu_file}
        COMMENT "Compiling Kunlun XPU kernel ${xpu_file}"
        VERBATIM
    )

    list(APPEND KUNLUN_XPU_OBJECT_FILES ${obj_file})
    list(APPEND KUNLUN_XPU_BIN_OBJECT_FILES ${bin_file})
endforeach()

add_custom_target(kunlun_xpu_kernels ALL DEPENDS
    ${KUNLUN_XPU_OBJECT_FILES} ${KUNLUN_XPU_BIN_OBJECT_FILES}
)
