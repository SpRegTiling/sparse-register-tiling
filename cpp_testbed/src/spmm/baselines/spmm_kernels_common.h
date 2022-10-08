//
// Created by lwilkinson on 5/25/22.
//

//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>

#include "utils/Vec.h"

#include "spmm.h"

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
inline __attribute__((always_inline)) int _a_row_times_b_col_panel_##TILE_SIZE_IN_VECTORS (                 \
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

//template<typename Scalar, typename Index, bool row_swizzle>
//__always_inline int _a_row_times_b_scalar(
//        int m, int k, int n,    // C (m x n) = A (m x k) * B (k x n)
//        int i, int j,
//        const int* __restrict__ row_indices,
//        const Scalar* __restrict__ values,
//        const Index* __restrict__ row_offsets,
//        const Index* __restrict__ column_indices,
//        const Scalar* __restrict__ B,
//        Scalar* __restrict__ C
//) {
//    for (; j < n; j ++) {
//        int row = i;
//        if constexpr(row_swizzle) { row = row_indices[i]; }
//
//        Scalar c_acc = C[(row * n) + j];
//
//        for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
//            c_acc += B[(column_indices[p] * n) + j] * values[p];
//        }
//
//        C[(row * n) + j] = c_acc;
//    }
//
//    return j;
//}

/******
 *      2D implementations
 *****/

#define A_TILE_TIMES_B_TILE_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                      \
template<typename VecType, typename StorageTypes, bool row_swizzle, bool load_c>                                \
int _a_tile_times_b_tile_##TILE_SIZE_IN_VECTORS (                                                               \
    int m, int k, int n,                                                                                        \
    int i, int ik, int j,                                                                                       \
    int mTile, int kTile,                                                                                       \
    const int* __restrict__ row_indices,                                                                        \
    const typename StorageTypes::Scalar* __restrict__ values,                                                   \
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,                                            \
    const typename StorageTypes::Index* __restrict__ column_indices,                                            \
    const typename StorageTypes::Scalar* __restrict__ B,                                                        \
    typename StorageTypes::Scalar* __restrict__ C,                                                              \
    int* nz_start_locs,                                                                                         \
    int* nz_end_locs                                                                                            \
) {                                                                                                             \
    int k_tile_end = ik + kTile;                                                                                \
    for (int ii = 0; ii < std::min(mTile, m - i); ii++) {                                                       \
        int row = ii + i, p;                                                                                    \
        if constexpr(row_swizzle) { row = row_indices[row]; }                                                   \
                                                                                                                \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_C, );                                                         \
        if constexpr(load_c) {                                                                                  \
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_C, );                                                    \
        } else {                                                                                                \
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, ZERO_C, );                                                    \
        }                                                                                                       \
                                                                                                                \
        for (p = nz_start_locs[ii]; p < row_offsets[row + 1]; ++p) {                                            \
            int b_row = column_indices[p];                                                                      \
            if (b_row >= k_tile_end) break;                                                                     \
            VecType aVec(values[p]);                                                                            \
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_B, );                                                     \
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_B, );                                                    \
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, FMA_BC, );                                                    \
        }                                                                                                       \
        nz_end_locs[ii] = p;                                                                                    \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, STORE_C, );                                                       \
    }                                                                                                           \
                                                                                                                \
    return j + TILE_SIZE_IN_VECTORS * VecType::size();                                                          \
}

A_TILE_TIMES_B_TILE_TEMPLATE(32)
A_TILE_TIMES_B_TILE_TEMPLATE(16)
A_TILE_TIMES_B_TILE_TEMPLATE(8)
A_TILE_TIMES_B_TILE_TEMPLATE(4)
A_TILE_TIMES_B_TILE_TEMPLATE(2)
A_TILE_TIMES_B_TILE_TEMPLATE(1)