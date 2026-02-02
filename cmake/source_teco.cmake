# ========================
# source_teco.cmake
# ========================
message(STATUS "Configuring TECO / Taichu backend")
add_compile_definitions(ENABLE_TECO_SDAA)

# ------------------------
# TECO_HOME
# ------------------------
if((NOT DEFINED TECO_HOME) AND (NOT DEFINED ENV{TECO_HOME}))
    set(TECO_HOME "/opt/tecoai")
    message(WARNING "TECO_HOME not defined, defaulting to /opt/tecoai")
elseif(DEFINED TECO_HOME)
    set(TECO_HOME ${TECO_HOME} CACHE STRING "TECO SDK directory")
else()
    set(TECO_HOME $ENV{TECO_HOME} CACHE STRING "TECO SDK directory")
endif()

message(STATUS "TECO_HOME: ${TECO_HOME}")

include_directories(
    ${TECO_HOME}/include
    ${CMAKE_SOURCE_DIR}/include
)

link_directories(${TECO_HOME}/lib64)

# ------------------------
# Source files
# ------------------------
file(GLOB TECO_SCPP_FILES "${CMAKE_SOURCE_DIR}/src/**/teco/*.scpp")
list(APPEND TECO_SOURCE_FILES ${TECO_SCPP_FILES})
set(PROJECT_TECO_SOURCES ${CPP_SOURCE_FILES})  # host cpp 文件

# ------------------------
# Compile .scpp -> .o
# ------------------------
set(SCPP_OBJECTS "")
set(TECO_OBJ_DIR "${CMAKE_BINARY_DIR}/teco_objs")
file(MAKE_DIRECTORY ${TECO_OBJ_DIR})

foreach(scpp_file ${TECO_SOURCE_FILES})
    get_filename_component(fname ${scpp_file} NAME_WE)
    set(obj_file "${TECO_OBJ_DIR}/${fname}.o")
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${TECO_HOME}/bin/tecocc ${scpp_file} -o ${obj_file}
            -O2 -fPIC -Wall -Werror -std=c++17 -pthread
            -I${TECO_HOME}/include
            -I${CMAKE_SOURCE_DIR}/include
        DEPENDS ${scpp_file}
        COMMENT "Compiling TECO source ${scpp_file}"
        VERBATIM
    )
    list(APPEND SCPP_OBJECTS ${obj_file})
endforeach()

add_custom_target(teco_objects ALL DEPENDS ${SCPP_OBJECTS})

# ------------------------
# TECO link libraries
# ------------------------
set(TECO_LINK_LIBS
    ${TECO_HOME}/lib64/libsdaart.so
    ${TECO_HOME}/lib64/libtecoblas.so
    ${TECO_HOME}/lib64/libtecodnn.so
    stdc++
)

