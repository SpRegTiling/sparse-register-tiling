CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)
project(nano-spmm-bench C CXX ASM)

# MSVC runtime library flags are selected by an abstraction.
CMAKE_POLICY(SET CMP0091 NEW)

INCLUDE(GNUInstallDirs)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Disable "note: parameter passing for argument of type ... changed/will change in ..."
    ADD_COMPILE_OPTIONS("-Wno-psabi")
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(spmm-nano-kernels)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${SOP_CXX_FLAGS})

add_subdirectory(third_party)
add_subdirectory(cpp_testbed)

