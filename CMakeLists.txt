cmake_minimum_required(VERSION 3.16)
project(MyCUDAProject)

# ========================
# Options
# ========================
option(USE_CUDA "Enable CUDA" OFF)
option(USE_DL "Enable DL" OFF)
option(USE_BANG "Enable BANG / Cambricon" OFF)
option(USE_KUNLUN "Enable Kunlun / XPU" OFF)
option(USE_ASCEND "Enable ascend" OFF)
option(USE_TECO "Enable TECO / Taichu" OFF)
option(USE_CPU "CPU only" OFF)

# ========================
# Common
# ========================
include(cmake/common.cmake)

set(ALL_SOURCES ${PROJECT_CPU_SOURCES})
set(ALL_LINK_LIBS "")

set(IS_NVIDIA_CUDA OFF)
set(IS_DLCC_CUDA OFF)
set(IS_BANG OFF)
set(IS_KUNLUN OFF)
set(IS_ASCEND OFF)
set(IS_TECO OFF)

# ========================
# CUDA / DLCC / NVIDIA
# ========================
if(USE_CUDA)
    set(IS_NVIDIA_CUDA ON)
    include(cmake/source_nvidia.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_CUDA_SOURCES})
    list(APPEND ALL_LINK_LIBS ${NVIDIA_LINK_LIBS})
endif()

if(USE_DL)
    # ---------- DLCC ----------
    set(IS_DLCC_CUDA ON)
    include(cmake/source_ql.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_QL_CPP_SOURCES})
    list(APPEND ALL_LINK_LIBS ${QL_LINK_LIBS})
endif()

# ========================
# BANG / Cambricon
# ========================
if(USE_BANG)
    set(IS_BANG ON)
    include(cmake/source_cambricon.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_BANG_SOURCES})
endif()

# ========================
# Kunlun / XPU
# ========================
if(USE_KUNLUN)
    set(IS_KUNLUN ON)
    include(cmake/source_kunlun.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_KUNLUN_SOURCES})
endif()

# ========================
# Ascend / NPU
# ========================
if(USE_ASCEND)
    set(IS_ASCEND ON)
    include(cmake/source_ascend.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_ASCEND_SOURCES})
endif()

if(USE_TECO)
    set(IS_TECO ON)
    include(cmake/source_teco.cmake)
    list(APPEND ALL_SOURCES ${PROJECT_TECO_SOURCES} ${SCPP_OBJECTS})
endif()
# ========================
# Create library
# ========================
if(IS_BANG)
    # BANG 平台只用 bang_add_library
    bang_add_library(my_library SHARED ${ALL_SOURCES})
    set_target_properties(my_library PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    target_link_libraries(my_library PRIVATE ${BANG_LINK_LIBS})

elseif(IS_KUNLUN)
    # Kunlun 平台
    add_library(my_library SHARED
        ${PROJECT_KUNLUN_SOURCES}       # host cpp
        ${KUNLUN_XPU_OBJECT_FILES}      # host .o
    )
    add_dependencies(my_library kunlun_xpu_kernels)

    # include paths
    target_include_directories(my_library PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${KUNLUN_HOME}/include
        ${XRE_DIR}/include
        ${XDNN_DIR}/include
        ${XTDK_DIR}/include
    )

    target_compile_options(my_library PRIVATE -Wall -fPIC)

    # 链接 device binary object，保证 kernel symbol 被包含
    target_link_libraries(my_library PRIVATE
        stdc++ xpuapi xpurt
        ${KUNLUN_XPU_BIN_OBJECT_FILES}
    )

    set_target_properties(my_library PROPERTIES LINKER_LANGUAGE CXX
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
elseif(IS_ASCEND)
    add_library(my_library SHARED ${PROJECT_ASCEND_SOURCES})
    set_target_properties(my_library PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    target_link_libraries(my_library PRIVATE ${ASCEND_LINK_LIBS})
elseif(IS_TECO)
    add_library(my_library SHARED
        ${PROJECT_TECO_SOURCES}
        ${SCPP_OBJECTS}  # TECO 编译的 .o
    )
    add_dependencies(my_library teco_objects)
    target_include_directories(my_library PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TECO_HOME}/include
    )
    target_compile_options(my_library PRIVATE -Wall -fPIC -std=c++17)
    target_link_libraries(my_library PRIVATE ${TECO_LINK_LIBS})
    set_target_properties(my_library PROPERTIES LINKER_LANGUAGE CXX
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
else()
    # NVIDIA / DLCC / CPU 平台
    add_library(my_library SHARED ${ALL_SOURCES})
    set_target_properties(my_library PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# ========================
# NVIDIA CUDA fixes
# ========================
if(IS_NVIDIA_CUDA)
    # 明确 CUDA 架构
    set_target_properties(my_library PROPERTIES
        CUDA_ARCHITECTURES 80
    )

    # CUTLASS 必需 flag
    target_compile_options(my_library PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# ========================
# DLCC CUDA fixes
# ========================
if(IS_DLCC_CUDA)
    # 添加 include path
    include_directories(${DLCC_CUDA_PATH}/include)
    target_include_directories(my_library PRIVATE ${DLCC_CUDA_PATH}/include)

    if(DEFINED CUTLASS_INCLUDE)
        include_directories(${CUTLASS_INCLUDE})
        target_include_directories(my_library PRIVATE ${CUTLASS_INCLUDE})
    endif()

    if(TARGET ql_cuda_objs)
        add_dependencies(my_library ql_cuda_objs)
        target_sources(my_library PRIVATE ${QL_CU_OBJECTS})
    endif()
endif()

# ========================
# Final linking
# ========================
target_link_libraries(my_library PRIVATE ${ALL_LINK_LIBS})
