
//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>
#include <mkl.h>

#include "utils/Vec.h"
#include "utils/misc.h"
#include "utils/error.h"

#include "SpMM_GECSB.h"


using std::vector;


using Config = GECSBConfig;


//#define INIT_C(z, _n, x)    VecType cVec##_n (0);
#define DEF_B(z, _n, x)     VecType bVec##_n;
#define DEF_C(z, _n, x)     VecType cVec##_n;
#define LOAD_B(z, _n, x)    bVec##_n.load(&B[(b_row * n) + j + _n * VecType::size()]);
#define LOAD_C(z, _n, x)    cVec##_n.load(&C[(row * n) + j + _n * VecType::size()]);
#define FMA_BC(z, _n, x)    cVec##_n = mul_add(aVec, bVec##_n, cVec##_n);
#define STORE_C(z, _n, x)   cVec##_n.store(&C[(row * n) + j + _n * VecType::size()]);

#define INNER_LOOP_CSR_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                        \
template<typename VecType, typename Scalar, typename PtrsType, enum GECSB_BLOCK_STORAGE storage, bool row_swizzle> \
static int _inner_loop_csr_tiled_##TILE_SIZE_IN_VECTORS(                                        \
    int m, int k, int n,                                                                                        \
    int j,                                                                                                      \
    int blk_i, int blk_k,                                                                                       \
    const int* __restrict__ row_indices,                                                                        \
    int blk_offset,                                                                                             \
    const PtrsType* __restrict__ b_row_ptrs,                                                                    \
    const uint16_t* __restrict__ b_column_indices,                                                              \
    const Scalar* __restrict__ values,                                                                          \
    const Scalar* __restrict__ B,                                                                               \
    Scalar* __restrict__ C,                                                                                     \
    const GECSBConfig& config                                                                                   \
) {                                                                                                             \
    int i_start = blk_i * config.mc_tile, i_end = (blk_i + 1) * config.mc_tile;                                 \
                                                                                                                \
    for (int i = i_start; i < i_end; i++, blk_offset++) {                                                       \
      int row = i;                                                                                              \
      if constexpr(row_swizzle) { row = row_indices[i]; }                                                       \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_C, );                                                           \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_C, );                                                          \
                                                                                                                \
      _Pragma("GCC unroll 2")                                                                                   \
      for (int p = b_row_ptrs[blk_offset]; p < b_row_ptrs[blk_offset + 1]; p++) {                               \
        int b_row = b_column_indices[p];                                                                        \
        VecType aVec(values[p]);                                                                                \
                                                                                                                \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_B, );                                                         \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_B, );                                                        \
        BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, FMA_BC, );                                                        \
      }                                                                                                         \
                                                                                                                \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, STORE_C, );                                                         \
    }                                                                                                           \
                                                                                                                \
    return j + VecType::size() * TILE_SIZE_IN_VECTORS;                                                          \
}

INNER_LOOP_CSR_TEMPLATE(32)
INNER_LOOP_CSR_TEMPLATE(16)
INNER_LOOP_CSR_TEMPLATE(8)
INNER_LOOP_CSR_TEMPLATE(4)
INNER_LOOP_CSR_TEMPLATE(2)
INNER_LOOP_CSR_TEMPLATE(1)

#define INNER_LOOP_CSC_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                           \
template<typename VecType, typename Scalar, typename PtrsType, enum GECSB_BLOCK_STORAGE storage, bool row_swizzle> \
static int _inner_loop_csc_tiled_##TILE_SIZE_IN_VECTORS(                                                        \
    int m, int k, int n,                                                                                        \
    int j,                                                                                                      \
    int blk_i, int blk_k,                                                                                       \
    const int* __restrict__ row_indices,                                                                        \
    int blk_offset,                                                                                             \
    const PtrsType* __restrict__ b_col_ptrs,                                                                    \
    const uint16_t* __restrict__ b_row_indices,                                                                 \
    const Scalar* __restrict__ values,                                                                          \
    const Scalar* __restrict__ B,                                                                               \
    Scalar* __restrict__ C,                                                                                     \
    const GECSBConfig& config                                                                                   \
) {                                                                                                             \
  int k_start = blk_k * config.kc_tile, k_end = (blk_k + 1) * config.kc_tile;                                   \
                                                                                                                \
  for (int ik = k_start; ik < std::min(k_end, k); ik++, blk_offset++) {                                         \
    int b_row = ik;                                                                                             \
                                                                                                                \
    BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_B, );                                                             \
    BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_B, );                                                            \
                                                                                                                \
    _Pragma("GCC unroll 2")                                                                                     \
    for (int p = b_col_ptrs[blk_offset]; p < b_col_ptrs[blk_offset + 1]; p++) {                                 \
      int row = b_row_indices[p];                                                                               \
      assert(row < m);                                                                                          \
      assert(ik < k);                                                                                           \
      if constexpr(row_swizzle) { row = row_indices[row]; }                                                     \
      VecType aVec(values[p]);                                                                                  \
                                                                                                                \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, DEF_C, );                                                           \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, LOAD_C, );                                                          \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, FMA_BC, );                                                          \
      BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, STORE_C, );                                                         \
    }                                                                                                           \
  }                                                                                                             \
                                                                                                                \
  return j + VecType::size() * TILE_SIZE_IN_VECTORS;                                                            \
}

INNER_LOOP_CSC_TEMPLATE(32)
INNER_LOOP_CSC_TEMPLATE(16)
INNER_LOOP_CSC_TEMPLATE(8)
INNER_LOOP_CSC_TEMPLATE(4)
INNER_LOOP_CSC_TEMPLATE(2)
INNER_LOOP_CSC_TEMPLATE(1)

/**
 *  Main Kernel
 */


#define MKL_CBLAS_TYPED_DISPATCH(func, ...)          \
    if constexpr(std::is_same_v<Scalar, float>)      \
        cblas_s##func (__VA_ARGS__);                 \
    else                                             \
        cblas_d##func (__VA_ARGS__);

template<typename PtrsType, typename Scalar, int vector_width, enum GECSB_BLOCK_STORAGE storage, int n_tile>
static void _kernel(
        int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
        const int* __restrict__       row_indices,
        const vector<vector<int>>&    block_ptrs,
        const PtrsType* __restrict__  b_ptrs,
        const uint16_t* __restrict__  b_inds,
        const Scalar* __restrict__    values,
        const Scalar* __restrict__    B,
        Scalar* __restrict__          C,
        const GECSBConfig& config
) {
    using VecType = typename Vec<Scalar, vector_width>::Type;

    int num_m_blks = std::ceil(m / double(config.mc_tile));
    int num_k_blks = std::ceil(k / double(config.kc_tile));
    int total_blks = num_m_blks * num_k_blks;
    using VecType = typename Vec<Scalar, vector_width>::Type;
    using ScalarType = Scalar;

    static constexpr int n_tile_in_vecs = n_tile / VecType::size();

    #define REGISTER_N_CSR_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs == (size_in_vecs)) \
            _inner_loop_csr_tiled_##size_in_vecs<VecType, Scalar, PtrsType, storage, false>( \
                m, k, n,                \
                j,                      \
                tii, tkk,               \
                row_indices,            \
                block_ptrs[tii][tkk],   \
                b_ptrs, b_inds, values, \
                B, C, config);

    #define REGISTER_N_CSC_TILE_SIZE(size_in_vecs) \
        if constexpr(n_tile_in_vecs == (size_in_vecs)) \
            _inner_loop_csc_tiled_##size_in_vecs<VecType, Scalar, PtrsType, storage, false>( \
                m, k, n,                \
                j,                      \
                tii, tkk,               \
                row_indices,            \
                block_ptrs[tii][tkk],   \
                b_ptrs, b_inds, values, \
                B, C, config);

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

    if (n % n_tile) {
        std::cerr << "TODO: fix cleanup code" << std::endl;
        exit(-1);
    }

    int Mb = std::ceil(m / double(config.mc_tile));
    int Nb = std::ceil(n / double(config.nc_tile));
    int Kb = std::ceil(k / double(config.kc_tile));
    int Nrb = std::ceil(config.nc_tile / config.nr_tile);

    #pragma omp parallel for schedule(static)
    for (int tii = 0; tii < Mb; tii++) {
      int  iii =  tii * config.mc_tile;
      for (int tkk = 0, kkk = 0; tkk < Kb; tkk++, kkk += config.kc_tile) {
        for (int tjj = 0, jjj = 0; tjj < Nb; tjj++, jjj += config.nc_tile) {
          for (int tj = 0, j = jjj; tj < Nrb; tj++, j += config.nr_tile) {
            if constexpr(storage == GECSB_CSR) {
              assert(j < n);

              CHECK_IS_IN(n_tile_in_vecs, 32, 16, 8, 4, 2, 1);
              REGISTER_N_CSR_TILE_SIZE(32);
              REGISTER_N_CSR_TILE_SIZE(16);
              REGISTER_N_CSR_TILE_SIZE(8);
              REGISTER_N_CSR_TILE_SIZE(4);
              REGISTER_N_CSR_TILE_SIZE(2);
              REGISTER_N_CSR_TILE_SIZE(1);
              // This loop increments j by += n_tile

              // TODO: Fix Code
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
            } else if constexpr(storage == GECSB_CSC) {

              CHECK_IS_IN(n_tile_in_vecs, 32, 16, 8, 4, 2, 1);
              REGISTER_N_CSC_TILE_SIZE(32);
              REGISTER_N_CSC_TILE_SIZE(16);
              REGISTER_N_CSC_TILE_SIZE(8);
              REGISTER_N_CSC_TILE_SIZE(4);
              REGISTER_N_CSC_TILE_SIZE(2);
              REGISTER_N_CSC_TILE_SIZE(1);
              // This loop increments j by += n_tile

              // TODO: Fix Code
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
        }
      }
    }
}

template<typename PtrsType, typename Scalar, int vector_width, enum GECSB_BLOCK_STORAGE storage>
static void _dispatch_n_tile_size(
        int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
        const int* __restrict__       row_indices,
        const vector<vector<int>>&    block_ptrs,
        const PtrsType* __restrict__  b_ptrs,
        const uint16_t* __restrict__  b_inds,
        const Scalar* __restrict__    b_values,
        const Scalar* __restrict__    B,
        Scalar* __restrict__          C,
        const GECSBConfig& config
) {
    switch (config.nr_tile) {
        case 16:
          _kernel<PtrsType, Scalar,  vector_width, storage, 16>(
                  m, k, n,
                  row_indices,
                  block_ptrs,
                  b_ptrs, b_inds, b_values,
                  B, C, config);
          return;
        case 32:
            _kernel<PtrsType, Scalar,  vector_width, storage, 32>(
                    m, k, n,
                    row_indices,
                    block_ptrs,
                    b_ptrs, b_inds, b_values,
                    B, C, config);
            return;
        case 64:
            _kernel<PtrsType, Scalar, vector_width, storage, 64>(
                    m, k, n,
                    row_indices,
                    block_ptrs,
                    b_ptrs, b_inds, b_values,
                    B, C, config);
            return;
        case 128:
            _kernel<PtrsType, Scalar, vector_width, storage, 128>(
                    m, k, n,
                    row_indices,
                    block_ptrs,
                    b_ptrs, b_inds, b_values,
                    B, C, config);
            return;
        // case 256:
        //     _kernel<PtrsType, Scalar, vector_width, storage, 256>(
        //             m, k, n,
        //             row_indices,
        //             block_ptrs,
        //             b_ptrs, b_inds, b_values,
        //             B, C, config);
        //     return;
        default:
            std::cerr << "Unsupported n_tile size" << std::endl;
            exit(-1);
    }
}

template<typename PtrsType, typename Scalar, int vector_width, enum GECSB_BLOCK_STORAGE storage>
void spmm_csb(
        int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
        const int* __restrict__       row_indices,
        const vector<vector<int>>&    block_ptrs,
        const PtrsType* __restrict__  b_ptrs,
        const uint16_t* __restrict__  b_inds,
        const Scalar* __restrict__    b_values,
        const Scalar* __restrict__    B,
        Scalar* __restrict__          C,
        int batch_size,
        const GECSBConfig& config
) {
    zero(C, m * n);

    _dispatch_n_tile_size<PtrsType, Scalar, vector_width, storage>(
            m, k, n,
            row_indices,
            block_ptrs,
            b_ptrs, b_inds, b_values,
            B, C, config);
}

#define _INSTANTIATE(RowPtrType, Scalar, vec_width, storage)              \
template void spmm_csb<RowPtrType, Scalar, vec_width, storage>(     \
    int m, int k, int _n, \
    const int* __restrict__         row_indices, \
    const vector<vector<int>>&      block_ptrs, \
    const RowPtrType* __restrict__  b_ptrs, \
    const uint16_t* __restrict__    b_inds, \
    const Scalar* __restrict__      b_values, \
    const Scalar* __restrict__      B, \
    Scalar* __restrict__            C, \
    int batch_size, \
    const GECSBConfig& config \
)

_INSTANTIATE(int,  float, 512, GECSB_CSR);
_INSTANTIATE(int,  float, 256, GECSB_CSR);

_INSTANTIATE(int,  float, 512, GECSB_CSC);
_INSTANTIATE(int,  float, 256, GECSB_CSC);

_INSTANTIATE(int,  double, 512, GECSB_CSR);
_INSTANTIATE(int,  double, 256, GECSB_CSR);

_INSTANTIATE(int,  double, 512, GECSB_CSC);
_INSTANTIATE(int,  double, 256, GECSB_CSC);
