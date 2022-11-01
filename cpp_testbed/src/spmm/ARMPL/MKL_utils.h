//
// Created by lwilkinson on 7/10/22.
//

#ifndef DNN_SPMM_BENCH_MKL_UTILS_H
#define DNN_SPMM_BENCH_MKL_UTILS_H

#include <mkl.h>

inline constexpr const char *mkl_error_string(sparse_status_t status) {
    switch (status) {
        case SPARSE_STATUS_SUCCESS: return "SPARSE_STATUS_SUCCESS";
        case SPARSE_STATUS_NOT_INITIALIZED: return "SPARSE_STATUS_NOT_INITIALIZED";
        case SPARSE_STATUS_ALLOC_FAILED: return "SPARSE_STATUS_ALLOC_FAILED";
        case SPARSE_STATUS_INVALID_VALUE: return "SPARSE_STATUS_INVALID_VALUE";
        case SPARSE_STATUS_EXECUTION_FAILED: return "SPARSE_STATUS_EXECUTION_FAILED";
        case SPARSE_STATUS_INTERNAL_ERROR: return "SPARSE_STATUS_INTERNAL_ERROR";
        case SPARSE_STATUS_NOT_SUPPORTED: return "SPARSE_STATUS_NOT_SUPPORTED";
        default: return "Unknown MKL Error";
    }
}

#define MKL_SPARSE_TYPED_DISPATCH(func, ...)         \
    if constexpr(std::is_same_v<Scalar, float>)      \
        status = mkl_sparse_s_##func (__VA_ARGS__);  \
    else                                             \
        status = mkl_sparse_d_##func (__VA_ARGS__);

#define MKL_CHECK(status)                           \
    if (status != SPARSE_STATUS_SUCCESS) {          \
        std::cerr << __FILE__ << ":" << __LINE__ << " " << mkl_error_string(status) << std::endl; exit(-2); \
    }


#endif //DNN_SPMM_BENCH_MKL_UTILS_H
