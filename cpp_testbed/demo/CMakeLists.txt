include(../cmake/LinkVtune.cmake)
include(../cmake/LinkCake.cmake)

##
#   Google Benchmark
##
SET(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
        CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
        CACHE PATH "Confu-style dependencies binary directory")

IF(NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(../cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
ENDIF()

SET(CMAKE_CXX_FLAGS_TEMP ${CMAKE_CXX_FLAGS})
SET(CMAKE_CXX_FLAGS "")

SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
ADD_SUBDIRECTORY(
        "${GOOGLEBENCHMARK_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_TEMP})


if (PAPI_FOUND)
    set(SPMM_DEMO_LIBS ${SPMM_DEMO_LIBS} ${PAPI_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPAPI_AVAILABLE")
endif()

set(SPMM_DEMO_INCLUDES ${CMAKE_CURRENT_LIST_DIR}/src)
list(APPEND SPMM_DEMO_INCLUDES ${PROJECT_SOURCE_DIR}/third_party)
list(APPEND SPMM_DEMO_INCLUDES ${CMAKE_CURRENT_LIST_DIR}/SpMM_methods)


aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SPMM_DEMO_SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/SpMM_methods" SPMM_SOURCES)

message(STATUS "SOP_SOURCES: ${SOP_SOURCES}")
message(STATUS "SPMM_METHODS_SOURCES: ${SPMM_METHODS_SOURCES}")

add_executable(SPMM_demo SPMM_demo.cpp
        ${SRC}
        ${SPMM_SOURCES}
        ${SPMM_METHODS_SOURCES}
        ${SPMM_DEMO_SRC}
        ${ASpT_SRCS}
        ${SOP_SOURCES}
        ${SOP_GENERATED_SRCS}
)

target_compile_options(SPMM_demo PRIVATE -save-temps=obj -funroll-loops ${SPMM_METHODS_CXX_FLAGS})
target_include_directories(SPMM_demo PUBLIC
        ${SPMM_DEMO_INCLUDES}
        ${SPMM_INCLUDE}
        ${SPMM_METHODS_INCLUDE_DIRS}
        ${MKL_INCLUDE_DIR}
        ${ASpT_INCLUDE_DIRS}
        ${SOP_ALL_INCLUDES}
)
target_link_libraries(SPMM_demo
        ${SPMM_METHODS_LIBRARIES}
        ${SPMM_DEMO_LIBS}
        -lstdc++fs
        benchmark
        ryml::ryml
) # stdc++fs for c++ filesystem
link_vtune_if_available(SPMM_demo)

#TARGET_INCLUDE_DIRECTORIES(x24-transpose-bench PRIVATE . include src)
#TARGET_LINK_LIBRARIES(x24-transpose-bench PRIVATE benchmark bench-utils cpuinfo fp16 pthreadpool)


#add_executable(SPMM_analyze SPMM_analyze.cpp ${SRC} ${SPMM_SOURCES})
#target_compile_options(SPMM_analyze PRIVATE -save-temps=obj -funroll-loops)
#target_include_directories(SPMM_analyze PUBLIC ${SPMM_INCLUDE} ${MKL_INCLUDE_DIR})
#target_link_libraries(SPMM_analyze ${SPMM_DEMO_LIBS} -lstdc++fs) # stdc++fs for c++ filesystem
#
#add_executable(u_kernel_testing u_kernel_testing.cpp ${SRC} ${SPMM_SOURCES} ${SPMM_DEMO_SRC} ${ASpT_SRCS})
#target_compile_options(u_kernel_testing PRIVATE -save-temps=obj -funroll-loops)
#target_include_directories(u_kernel_testing PUBLIC ${SPMM_DEMO_INCLUDES} ${SPMM_INCLUDE} ${MKL_INCLUDE_DIR} ${ASpT_INCLUDE_DIRS})
#target_link_libraries(u_kernel_testing ${SPMM_DEMO_LIBS} -lstdc++fs ryml::ryml) # stdc++fs for c++ filesystem
#
#add_executable(coo_block_test coo_block_test.cpp ${SRC} ${SPMM_SOURCES} ${SPMM_DEMO_SRC} ${ASpT_SRCS})
#target_compile_options(coo_block_test PRIVATE -save-temps=obj -funroll-loops)
#target_include_directories(coo_block_test PUBLIC ${SPMM_DEMO_INCLUDES} ${SPMM_INCLUDE} ${MKL_INCLUDE_DIR} ${ASpT_INCLUDE_DIRS})
#target_link_libraries(coo_block_test ${SPMM_DEMO_LIBS} -lstdc++fs ryml::ryml) # stdc++fs for c++ filesystem
#
#
#if (PAPI_FOUND)
#    add_executable(Profiler_test Profiler_test.cpp ${SRC})
#    target_compile_options(Profiler_test PRIVATE -save-temps=obj -funroll-loops)
#    target_link_libraries(Profiler_test ${SPMM_DEMO_LIBS}) # stdc++fs for c++ filesystem
#endif()
