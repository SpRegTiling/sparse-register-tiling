//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>

#include "utils/Vec.h"
#include "utils/misc.h"
#include "utils/error.h"

#include "spmm.h"
#include "spmm_kernels_common.h"

using Config = CSR_C_Config;

//template<typename Scalar, typename Index, bool row_swizzle>
//__always_inline int _inner_loop_scalar(
//        int m, int k, int n,    // C (m x n) = A (m x k) * B (k x n)
//        int i, int j,
//        const int* __restrict__ row_indices,
//        const Scalar* __restrict__ values,
//        const Index* __restrict__ row_offsets,
//        const Index* __restrict__ column_indices,
//        const Scalar* __restrict__ B,
//        Scalar* __restrict__ C
//) {
//    j = _a_row_times_b_scalar<Scalar, row_swizzle>(
//        m, k, n,    // C (m x n) = A (m x k) * B (k x n)
//        i, j,
//        row_indices, values, row_offsets, column_indices,
//        B, C
//    );
//
//    return j;
//}

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
    typename StorageTypes::Scalar* __restrict__ C                                                               \
) {                                                                                                             \
    j = _a_row_times_b_col_panel_##TILE_SIZE_IN_VECTORS<VecType, StorageTypes, row_swizzle, false>(             \
        m, k, n,                                                                                                \
        i, j,                                                                                                   \
        row_indices, values, row_offsets, column_indices,                                                       \
        B, C                                                                                                    \
    );                                                                                                          \
                                                                                                                \
    return j;                                                                                                   \
}

INNER_LOOP_CSR_TEMPLATE(32)
INNER_LOOP_CSR_TEMPLATE(16)
INNER_LOOP_CSR_TEMPLATE(8)
INNER_LOOP_CSR_TEMPLATE(4)
INNER_LOOP_CSR_TEMPLATE(2)
INNER_LOOP_CSR_TEMPLATE(1)

template<typename StorageTypes, int vector_width, bool row_swizzle, int n_tile>
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
    using VecType = typename Vec<float, vector_width>::Type;
    using ScalarType = float;

    static constexpr int n_tile_in_vecs = n_tile / VecType::size();

#define REGISTER_N_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs == (size_in_vecs)) \
            j = _inner_loop_tiled_##size_in_vecs<VecType, StorageTypes, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C)

#define CLEAN_UP_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs > size_in_vecs) \
            j = _inner_loop_tiled_##size_in_vecs<VecType, ScalarType, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C)

#define CLEAN_UP_VECWIDTH(reduced_vec_width) \
        if constexpr(vector_width > reduced_vec_width) \
            j = _inner_loop_tiled_1< typename Vec<ScalarType, reduced_vec_width>::Type, ScalarType, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C);

    if (n % n_tile) {
        std::cerr << "TODO: fix cleanup code" << std::endl;
        exit(-1);
    }

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m; i ++) {
        int j = 0;

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
//                m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C);
    }
}

/**
 *  Main Kernel
 */
template<typename StorageTypes, int vector_width>
void spmm_csr_c(
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
    ERROR_AND_EXIT_IF(row_indices != nullptr, "row_indices != nullptr");

    switch (config.nrTile) {
      case 16:
        _dispatch_n_tile_size<StorageTypes, vector_width, false, 16>(
            m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
        return;
      case 32:
        _dispatch_n_tile_size<StorageTypes, vector_width, false, 32>(
            m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
        return;
      case 64:
        _dispatch_n_tile_size<StorageTypes, vector_width, false, 64>(
            m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
        return;
      case 128:
        _dispatch_n_tile_size<StorageTypes, vector_width, false, 128>(
            m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
        return;
      case 256:
        _dispatch_n_tile_size<StorageTypes, vector_width, false, 256>(
            m, k, n, row_indices, values, row_offsets, column_indices, B, C, config);
        return;
      default:
        std::cerr << "Unsupported n_tile size" << std::endl;
        exit(-1);
    }
}

INSTANTIATE_FOR_ALL_STORAGE_CONFIGS(spmm_csr_c, Config);
