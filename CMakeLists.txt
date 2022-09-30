cmake_minimum_required(VERSION 3.14.7)
project(dnn-spmm-bench)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/opt/miniconda/envs/xformer/lib/python3.8/site-packages/torch/include)
include_directories(/opt/miniconda/envs/xformer/lib/python3.8/site-packages/torch/include/torch/csrc/api/include)
include_directories(/opt/miniconda/envs/xformer/lib/python3.8/site-packages/torch/include/TH)
include_directories(/opt/miniconda/envs/xformer/lib/python3.8/site-packages/torch/include/THC)

add_subdirectory(SOP)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${SOP_CXX_FLAGS})

add_link_options(-lstdc++)
add_subdirectory(xformers/sparse/backends/cpu/spmm/)
add_subdirectory(third_party)
add_subdirectory(cpp_testbed)

