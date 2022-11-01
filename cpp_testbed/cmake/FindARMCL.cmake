set(ARMCL_INC_PATHS
        /usr/include
        /usr/local/include
        /usr/local/acl
        $ENV{ARMCL_DIR}
        $ENV{ARMCL_DIR}/include
        )

set(ARMCL_LIB_PATHS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/local/acl/lib
        /usr/local/acl/lib64
        $ENV{ARMCL_DIR}/lib
        $ENV{ARMCL_DIR}/build
        $ENV{ARMCL_DIR}/lib/linux-arm64-v8a-cl
        $ENV{ARMCL_DIR}/lib/arm64-v8a-neon
        )

find_path(ARMCL_INCLUDE NAMES arm_compute PATHS ${ARMCL_INC_PATHS})
#find_library(ARMCL_LIBRARIES NAMES arm_compute-static PATHS ${ARMCL_LIB_PATHS})
#find_library(ARMCL_CORE_LIBRARIES NAMES arm_compute_core-static PATHS ${ARMCL_LIB_PATHS})

find_library(ARMCL_LIBRARIES NAMES arm_compute PATHS ${ARMCL_LIB_PATHS})
find_library(ARMCL_CORE_LIBRARIES NAMES arm_compute_core PATHS ${ARMCL_LIB_PATHS})
find_library(ARMCL_GRAPH_LIBRARIES NAMES arm_compute_graph PATHS ${ARMCL_LIB_PATHS})

SET(ARMCL_LIBRARIES ${ARMCL_CORE_LIBRARIES} ${ARMCL_LIBRARIES} ${ARMCL_GRAPH_LIBRARIES})

if(ARMCL_INCS)
    SET(ARMCL_INCLUDE "${ARMCL_INCS}")
    SET(ARMCL_LIBRARIES "${ARMCL_LIBS}")
    SET(ARMCL_FOUND 1)
else  ()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(ARMCL DEFAULT_MSG ARMCL_INCLUDE ARMCL_LIBRARIES)
endif ()

if (ARMCL_FOUND)
    message(STATUS "Found ARMCL    (include: ${ARMCL_INCLUDE}, library: ${ARMCL_LIBRARIES})")
    mark_as_advanced(ARMCL_INCLUDE ARMCL_LIBRARIES)
endif ()