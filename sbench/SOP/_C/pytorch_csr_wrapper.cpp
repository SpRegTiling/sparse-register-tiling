//
// Created by lwilkinson on 7/20/22.
//

#include <assert.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <torch/script.h>
#include <torch/types.h>
#include <torch/custom_class.h>

#include <torch/extension.h>
#include <torch/library.h>

#include "utils/misc.h"

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>
#include "vec_type_utils.h"

template<typename _Scalar, typename _IndexPtr, typename _Index>
struct StorageTypes {
    using Scalar = _Scalar;
    using IndexPtr = _IndexPtr;
    typedef _Index Index;
};


/**
  Use the C preprocessor to splat out implementations with varying tile width of (this if for tile of 2 vectors):
    for (; j < n - (VecType::size() * 2 - 1); j += VecType::size() * 2) {
        VecType cVec0(0), cVec1(0);

        for (int p = row_offsets[i]; p < row_offsets[i + 1]; ++p) {
            VecType aVec(values[p]);
            int b_row = column_indices[p];
            VecType bVec0, bVec1;

            bVec.load(&B[(b_row * n) + j + 0 * VecType::size()]);
            bVec.load(&B[(b_row * n) + j + 1 * VecType::size()]);
            cVec0 = mul_add(aVec, bVec0, cVec0);
            cVec1 = mul_add(aVec, bVec1, cVec1);
        }

        cVec0.store(&C[(i * n) + j + 0 * VecType::size()]);
        cVec1.store(&C[(i * n) + j + 1 * VecType::size()]);
    }
    return j;
 */


/******
 *    1D implementations
 ******/

#define DEF_C(z, _n, x)     VecType cVec##_n;
#define ZERO_C(z, _n, x)    cVec##_n = VecType(0);
#define LOAD_C(z, _n, x)    cVec##_n.load(&C[(row * n) + j + _n * VecType::size()]);
#define DEF_B(z, _n, x)     VecType bVec##_n;
#define LOAD_B(z, _n, x)    bVec##_n.load(&B[(b_row * n) + j + _n * VecType::size()]);
#define FMA_BC(z, _n, x)    cVec##_n = mul_add(aVec, bVec##_n, cVec##_n);
#define STORE_C(z, _n, x)   cVec##_n.store(&C[(row * n) + j + _n * VecType::size()]);

#define A_ROW_TIMES_B_COL_PANEL_TEMPLATE(TILE_SIZE_IN_VECTORS)                                              \
template<typename VecType, typename StorageTypes, bool row_swizzle, bool load_c>                            \
__always_inline int _a_row_times_b_col_panel_##TILE_SIZE_IN_VECTORS (                                       \
    int m, int k, int n,                                                                                    \
    int i, int j,                                                                                           \
    const int* __restrict__ row_indices,                                                                    \
    const typename StorageTypes::Scalar* __restrict__ values,                                               \
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,                                        \
    const typename StorageTypes::Index* __restrict__ column_indices,                                        \
    const typename StorageTypes::Scalar* __restrict__ B,                                                    \
    typename StorageTypes::Scalar* __restrict__ C                                                           \
) {                                                                                                         \
    int row = i;                                                                                            \
    if constexpr(row_swizzle) { row = row_indices[i]; }                                                     \
                                                                                                            \
    BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_C, );                                                         \
    if constexpr(load_c) {                                                                                  \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_C, );                                                    \
    } else {                                                                                                \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, ZERO_C, );                                                    \
    }                                                                                                       \
                                                                                                            \
    for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {                                         \
        VecType aVec(values[p]);                                                                            \
        int b_row = column_indices[p];                                                                      \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_B, );                                                     \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_B, );                                                    \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, FMA_BC, );                                                    \
    }                                                                                                       \
    BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, STORE_C, );                                                       \
                                                                                                            \
    return j + TILE_SIZE_IN_VECTORS * VecType::size();                                                      \
}

A_ROW_TIMES_B_COL_PANEL_TEMPLATE(32)
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(16)
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(8)
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(4)
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(2)
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(1)

#define INNER_LOOP_CSR_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                           \
template<typename VecType, typename StorageTypes, bool row_swizzle>                                             \
static __always_inline int _inner_loop_tiled_##TILE_SIZE_IN_VECTORS (                                           \
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
        typename StorageTypes::Scalar* __restrict__ C
) {
    using VecType = typename Vec<float, vector_width>::Type;
    using ScalarType = float;

    static constexpr int n_tile_in_vecs = n_tile / VecType::size();

#define REGISTER_N_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs == (size_in_vecs)) \
            j = _inner_loop_tiled_##size_in_vecs<VecType, StorageTypes, row_swizzle>( \
                    m, k, n, i, j, row_indices, values, row_offsets, column_indices, B, C)

    if (n % n_tile) {
        std::cerr << "TODO: fix cleanup code" << std::endl;
        exit(-1);
    }

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

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SIZE_(x, y, z) TORCH_CHECK(x.size(y) == z, #x " size mismatch for " #z)
#define CHECK_SIZE(x, y) TORCH_CHECK(x == y, #x " size mismatch with " #y)

template<typename T, typename V, int n_tile>
double profile(
    uint32_t m, uint32_t k, uint32_t n,
    const V* __restrict__ values,
    const T* __restrict__ row_offsets,
    const T* __restrict__ column_indices,
    const V* __restrict__ B,
    V* __restrict__ C,
    int64_t num_runs
) {
    static int MEASURED_ITERATIONS = 150;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;


    using CSRStorageType = StorageTypes<V, T, T>;
    // Warmup runs
    for(int runs = 0; runs < num_runs; runs++) {
        _dispatch_n_tile_size<CSRStorageType, 512, false, n_tile>(
            m, k, n,
            nullptr,
            values, row_offsets, column_indices,
            B, C
        );
    }

    std::vector<double> timings(MEASURED_ITERATIONS);

    for (int iter = 0; iter < MEASURED_ITERATIONS; iter++) {
        start_time = std::chrono::high_resolution_clock::now();
        for (int runs = 0; runs < num_runs; runs++) {
            _dispatch_n_tile_size<CSRStorageType, 512, false, n_tile>(
                    m, k, n,
                    nullptr,
                    values, row_offsets, column_indices,
                    B, C
            );
        }
        end_time = std::chrono::high_resolution_clock::now();
        timings[iter] = (end_time - start_time).count();
    }

    zero(C, m * n);

    return median(timings) / double(num_runs);
}

template<typename T, typename V>
double _csr_kernel(
        T* row_ptrs, T* col_inds, V* values, V* B, V* C,
        int64_t rows, int64_t cols, int64_t bCols,
        int64_t num_runs,
        int n_tile
) {
    zero(C, rows * bCols);

    double timing;

    switch (n_tile) {
        case 16:
            timing = profile<T, V, 16>(rows, cols, bCols, values, row_ptrs, col_inds, B, C, num_runs);
            break;
        case 32:
            timing = profile<T, V, 32>(rows, cols, bCols, values, row_ptrs, col_inds, B, C, num_runs);
            break;
        case 64:
            timing = profile<T, V, 64>(rows, cols, bCols, values, row_ptrs, col_inds, B, C, num_runs);
            break;
        case 128:
            timing = profile<T, V, 128>(rows, cols, bCols, values, row_ptrs, col_inds, B, C, num_runs);
            break;
    }


    zero(C, rows * bCols);
    using CSRStorageType = StorageTypes<V, T, T>;

    switch (n_tile) {
        case 16:
            _dispatch_n_tile_size<CSRStorageType, 512, false, 16>(
                rows, cols, bCols,
                nullptr,
                values, row_ptrs, col_inds,
                B, C
            );
            break;
        case 32:
            _dispatch_n_tile_size<CSRStorageType, 512, false, 32>(
                rows, cols, bCols,
                nullptr,
                values, row_ptrs, col_inds,
                B, C
            );
            break;
        case 64:
            _dispatch_n_tile_size<CSRStorageType, 512, false, 64>(
                rows, cols, bCols,
                nullptr,
                values, row_ptrs, col_inds,
                B, C
            );
            break;
        case 128:
            _dispatch_n_tile_size<CSRStorageType, 512, false, 128>(
                rows, cols, bCols,
                nullptr,
                values, row_ptrs, col_inds,
                B, C
            );
            break;
    }

//    free(buffer);
//    delete[] panel_descs;

    return  timing;
}

std::tuple<double, torch::Tensor> executor(
    const at::sparse_csr::SparseCsrTensor& mat,
    const torch::Tensor B,
    int64_t num_runs,
    int64_t n_tile
) {
    TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
    TORCH_CHECK(mat.dim() == 2);

    auto nnz = mat._nnz();
    auto rows = mat.size(0);
    auto cols = mat.size(1);
    auto bCols = B.size(1);

    CHECK_SIZE(mat.size(1), B.size(0));
    CHECK_CONTIGUOUS(B);

    auto col_indices = mat.col_indices();
    auto row_ptrs = mat.crow_indices();
    auto values = mat.values();

    auto C = torch::zeros({ rows, bCols }, torch::kFloat32);

    double ret;

    AT_DISPATCH_INTEGRAL_TYPES(col_indices.scalar_type(), "_csr_kernel", ([&] {
        ret = _csr_kernel(
            row_ptrs.data_ptr<scalar_t>(),
            col_indices.data_ptr<scalar_t>(),
            values.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            rows, cols, bCols,
            num_runs,
            n_tile
        );
    }));

    return { ret, C };
}

#ifdef USE_JIT
PYBIND11_MODULE(csr, m) {
#else
TORCH_LIBRARY(csr, m) {
#endif
    m.def("executor", executor);
}