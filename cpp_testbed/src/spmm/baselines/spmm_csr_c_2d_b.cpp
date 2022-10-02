//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>
#include <algorithm>
#include <iostream>
#include <omp.h>

#include "utils/misc.h"
#include "utils/Vec.h"

#include "spmm.h"
#include "spmm_kernels_common.h"

using Config = CSR_C_2D_B_Config;


#define INNER_LOOP_CSR_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                           \
template<typename VecType, typename StorageTypes, bool row_swizzle>                                             \
static int _inner_loop_tiled_##TILE_SIZE_IN_VECTORS (                                                           \
    int m, int k, int n,                                                                                        \
    int i, int j,                                                                                               \
    const int* __restrict__ row_indices,                                                                        \
    const typename StorageTypes::Scalar* __restrict__ values,                                                   \
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,                                            \
    const typename StorageTypes::Index* __restrict__ column_indices,                                            \
    const typename StorageTypes::Scalar* __restrict__ B,                                                        \
    typename StorageTypes::Scalar* __restrict__ C,                                                              \
    int* __restrict__ nz_loc_workspace,                                                                         \
    const Config& config                                                                                        \
) {                                                                                                             \
    for (int ii = 0; ii < std::min(config.mTile, m - i); ii++) {                                                \
        int row = ii + i;                                                                                       \
        if constexpr(row_swizzle) { row = row_indices[row]; }                                                   \
        nz_loc_workspace[ii] = row_offsets[row];                                                                \
    }                                                                                                           \
                                                                                                                \
    for (int ik = 0; ik < k; ik += config.kTile) {                                                              \
        _a_tile_times_b_tile_##TILE_SIZE_IN_VECTORS <VecType, StorageTypes, row_swizzle, true> (                \
            m, k, n,                                                                                            \
            i, ik, j,                                                                                           \
            config.mTile, config.kTile,                                                                         \
            row_indices, values, row_offsets, column_indices,                                                   \
            B, C,                                                                                               \
            (int*) nz_loc_workspace,                                                                            \
            (int*) nz_loc_workspace                                                                             \
        );                                                                                                      \
    }                                                                                                           \
                                                                                                                \
    return j + TILE_SIZE_IN_VECTORS * VecType::size();                                                          \
}

INNER_LOOP_CSR_TEMPLATE(32)
INNER_LOOP_CSR_TEMPLATE(16)
INNER_LOOP_CSR_TEMPLATE(8)
INNER_LOOP_CSR_TEMPLATE(4)
INNER_LOOP_CSR_TEMPLATE(2)
INNER_LOOP_CSR_TEMPLATE(1)

//int next_nz_loc_workspace[config.mTile];                                                                    \

///**
// *  Code for handling residuals
// */
//template<typename Scalar, bool row_swizzle>
//static inline int _inner_loop_scalar(
//    int m, int k, int n,    // C (m x n) = A (m x k) * B (k x n)
//    int i, int j,
//    const int* __restrict__ row_indices,
//    const float* __restrict__ values,
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const float* __restrict__ B,
//    float* __restrict__ C,
//    const Config& config
//) {
//    for (; j < n; j ++) {
//        for (int ii = i; ii < std::min(i + config.mTile, k); ii++) {
//            int row = ii;
//            if constexpr(row_swizzle) { row = row_indices[ii]; }
//
//            Scalar c_acc = C[(row * n) + j];
//
//            for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
//                c_acc += B[(column_indices[p] * n) + j] * values[p];
//            }
//
//            C[(row * n) + j] = c_acc;
//        }
//    }
//
//    return j;
//}


template<typename StorageTypes, int vector_width, bool row_swizzle, int n_tile>
static void _kernel(
        int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
        const int* __restrict__ row_indices,
        const typename StorageTypes::Scalar* __restrict__ values,
        const typename StorageTypes::IndexPtr* __restrict__ row_offsets,
        const typename StorageTypes::Index* __restrict__ column_indices,
        const typename StorageTypes::Scalar* __restrict__ B,
        typename StorageTypes::Scalar* __restrict__ C,
        const Config& config
) {
    using VecType = typename Vec<float, vector_width>::Type;
    using ScalarType = float;

    static constexpr int n_tile_in_vecs = n_tile / VecType::size();

#define REGISTER_N_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs == (size_in_vecs)) \
            j = _inner_loop_tiled_##size_in_vecs<VecType, StorageTypes, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, \
                    B, C, local_workspace, config)

#define CLEAN_UP_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs > size_in_vecs) \
            j = _inner_loop_tiled_##size_in_vecs<VecType, ScalarType, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, \
                    B, C, local_workspace, config)

#define CLEAN_UP_VECWIDTH(reduced_vec_width) \
        if constexpr(vector_width > reduced_vec_width) \
            j = _inner_loop_tiled_1< typename Vec<ScalarType, reduced_vec_width>::Type, ScalarType, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices,                          \
                    B, C, local_workspace, config);

    static int workspace[20 * 1046];

    if (n % n_tile) {
        std::cerr << "TODO: fix cleanup code" << std::endl;
        exit(-1);
    }

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m; i += config.mTile) {
        int j = 0;
        int tid = 0; // omp_get_thread_num();
        int* local_workspace = &workspace[tid * 1046];

        while (j < n - (n_tile - 1)) {
            CHECK_IS_IN(n_tile_in_vecs, 32, 16, 8, 4, 2, 1);
            REGISTER_N_TILE_SIZE(32);
            REGISTER_N_TILE_SIZE(16);
            REGISTER_N_TILE_SIZE(8);
            REGISTER_N_TILE_SIZE(4);
            REGISTER_N_TILE_SIZE(2);
            REGISTER_N_TILE_SIZE(1);
            // This loop increments j by += n_tile
        }

// TODO: Fix speed-up
//        CLEAN_UP_TILE_SIZE(16);
//        CLEAN_UP_TILE_SIZE(8);
//        CLEAN_UP_TILE_SIZE(4);
//        CLEAN_UP_TILE_SIZE(2);
//        CLEAN_UP_TILE_SIZE(1);
//
//        CLEAN_UP_VECWIDTH(256);
//
//        _inner_loop_scalar<ScalarType, row_swizzle>(
//                m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C, config);
    }
}

template<typename StorageTypes, int vector_width, bool row_swizzle>
static void _dispatch_n_tile_size(
        int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
        const int* __restrict__ row_indices,
        const typename StorageTypes::Scalar* __restrict__ values,
        const typename StorageTypes::IndexPtr* __restrict__ row_offsets,
        const typename StorageTypes::Index* __restrict__ column_indices,
        const typename StorageTypes::Scalar* __restrict__ B,
        typename StorageTypes::Scalar* __restrict__ C,
        const Config& config
) {
    switch (config.nTile) {
        case 32:
            _kernel<StorageTypes, vector_width, row_swizzle, 32>(
                    m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
            return;
        case 64:
            _kernel<StorageTypes, vector_width, row_swizzle, 64>(
                    m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
            return;
        case 128:
            _kernel<StorageTypes, vector_width, row_swizzle, 128>(
                    m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
            return;
        case 256:
            _kernel<StorageTypes, vector_width, row_swizzle, 256>(
                    m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
            return;
        default:
            std::cerr << "Unsupported n_tile size" << std::endl;
            exit(-1);
    }
}

/**
 *  Main Kernel
 */
template<typename StorageTypes, int vector_width>
void spmm_csr_c_2d_b(
    int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
    int nonzeros,
    const int* __restrict__ row_indices,
    const typename StorageTypes::Scalar* __restrict__ values,
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,
    const typename StorageTypes::Index* __restrict__ column_indices,
    const typename StorageTypes::Scalar* __restrict__ B,
    typename StorageTypes::Scalar* __restrict__ C,
    int batch_size,
    const Config& config
) {
    assert(batch_size == 1);
    zero(C, m*n);

    if (row_indices == nullptr) {
        _dispatch_n_tile_size<StorageTypes, vector_width, false>(
                m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
    } else {
        _dispatch_n_tile_size<StorageTypes, vector_width, true>(
                m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
    }
}

INSTANTIATE_FOR_ALL_STORAGE_CONFIGS(spmm_csr_c_2d_b, Config);
