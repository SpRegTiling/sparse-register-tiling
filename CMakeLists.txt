CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)
project(nano-spmm-bench C CXX ASM)

INCLUDE(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(spmm-nano-kernels)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${SOP_CXX_FLAGS})

add_subdirectory(third_party)
add_subdirectory(cpp_testbed)

