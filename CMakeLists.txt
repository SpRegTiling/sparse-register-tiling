cmake_minimum_required(VERSION 3.14.7)
project(nano-spmm-bench)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

macro(remove_cxx_flag flag)
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endmacro()

#remove_cxx_flag("-O3")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -Wregister")

add_subdirectory(spmm-nano-kernels)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${SOP_CXX_FLAGS})

add_subdirectory(third_party)
add_subdirectory(cpp_testbed)

