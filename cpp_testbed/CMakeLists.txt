set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -std=c++2a -ffast-math")


if (${RASPBERRY_PI})
    ##
    # ARM
    ##
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ARM_ARCH_8__ -mfpu=neon-fp-armv8 -mcpu=cortex-a72")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRASPBERRY_PI=4")
    set(TUNE cortex-a72)
else()
    ##
    # Intel
    ##
    set(TUNE intel)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mprefer-vector-width=512")
endif()


## GCC
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=${TUNE} ${CMAKE_ARCH_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(STATUS "Using intel compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DC4_CPP=20 -DC4_CPP20")
    # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
endif()

set(MIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/")

find_package(METIS OPTIONAL_COMPONENTS)
find_package(PAPI OPTIONAL_COMPONENTS)

add_definitions(-DUSE_GLOP)

if(RYML_BRANCH_NAME STREQUAL "")
    find_program(GIT git REQUIRED)
    execute_process(COMMAND git rev-parse --abbrev-ref HEAD
            OUTPUT_VARIABLE RYML_BRANCH_NAME
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(RYML_BRANCH_NAME STREQUAL HEAD)
    set(RYML_BRANCH_NAME master)
endif()
message(STATUS "FetchContent from branch: ${RYML_BRANCH_NAME}")

include(FetchContent)
FetchContent_Declare(ryml
    GIT_REPOSITORY https://github.com/biojppm/rapidyaml.git
    GIT_TAG ${RYML_BRANCH_NAME}
    GIT_SHALLOW FALSE  # ensure submodules are checked out
)
FetchContent_MakeAvailable(ryml)

if (PAPI_FOUND)
    add_definitions(-DPAPI)
endif()

if ($ENV{PROFILE})
    add_definitions(-DPROFILE)
    include_directories(${PAPI_INCLUDE_DIRS})
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}/../SOP/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/row_reordering")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/row_reordering/distance_functions")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")

aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/row_reordering" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/row_reordering/distance_functions" SRC)

add_subdirectory(src)
add_subdirectory(demo)