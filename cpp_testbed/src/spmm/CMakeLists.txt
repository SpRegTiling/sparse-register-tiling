set(SPMM_METHODS_SOURCES "${CMAKE_CURRENT_LIST_DIR}/runtime_yaml_config.cpp")
set(SPMM_METHODS_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
set(SPMM_METHODS_CXX_FLAGS "")


set(SPMM_ASpT       ON)
set(SPMM_MKL        ON)
set(SPMM_baselines  ON)
set(SPMM_nano       ON)
set(SPMM_GECSB      ON)

macro(check_spmm_method NAME)
    IF(SPMM_${NAME})
        add_subdirectory(${NAME})
        list(APPEND SPMM_METHODS_CXX_FLAGS -DSPMM_${NAME}_ENABLED=1)
        MESSAGE(STATUS "SPMM METHOD: ${NAME} Enabled")
    ENDIF()
endmacro()


check_spmm_method(ASpT)
check_spmm_method(MKL)
check_spmm_method(baselines)
check_spmm_method(GECSB)
check_spmm_method(nano)

message(STATUS "SPMM_METHODS_SOURCES: ${SPMM_METHODS_SOURCES}")
message(STATUS "SPMM_METHODS_INCLUDE_DIRS: ${SPMM_METHODS_INCLUDE_DIRS}")

set(SPMM_METHODS_SOURCES ${SPMM_METHODS_SOURCES} PARENT_SCOPE)
set(SPMM_METHODS_INCLUDE_DIRS ${SPMM_METHODS_INCLUDE_DIRS} PARENT_SCOPE)
set(SPMM_METHODS_CXX_FLAGS ${SPMM_METHODS_CXX_FLAGS} PARENT_SCOPE)