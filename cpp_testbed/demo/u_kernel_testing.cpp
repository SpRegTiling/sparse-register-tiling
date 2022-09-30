//
// Created by lwilkinson on 6/23/22.
//

#include <assert.h>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <vectorclass.h>
#include "Vec.h"
#include "dense_kernels.hpp"
#include "math.h"
#include "spmm.h"
#include "utils.h"

struct aux_s {
    double *b_next;
    float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};

typedef struct aux_s aux_t;

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
    for (; i < m; i ++) {                                                                                   \
        int row = i;                                                                                            \
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
    }                                                                                                       \
    return j + TILE_SIZE_IN_VECTORS * VecType::size();                                                      \
}


A_ROW_TIMES_B_COL_PANEL_TEMPLATE(1);
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(2);
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(4);
A_ROW_TIMES_B_COL_PANEL_TEMPLATE(8);


struct COOBlock {
    int* col_inds;
    int* row_inds;
    float* values;
};

struct CSRBlock {
    int* row_ptrs;
    int* col_inds;
    float* values;
};

struct CSCBlock {
    int* row_ptrs;
    int* col_inds;
    float* values;
};

struct COOBlock construct_sample_coo_block(int m, int n) {
    struct COOBlock block;

    block.col_inds = new int[m];
    block.row_inds = new int[n];
    block.values = new float[m * n];

    for (int i = 0; i < m; i++) block.col_inds[i] = i;
    for (int i = 0; i < n; i++) block.row_inds[i] = i;
    for (int i = 0; i < m*n; i++) block.values[i] = (float) rand() / (float) (RAND_MAX);

    return block;
}

struct CSRBlock construct_sample_csr_block(int m, int n, int nnz) {
    struct CSRBlock block;

    //int nnz = std::ceil((m*n) * sparsity);

    block.row_ptrs = new int[m + 1];
    block.col_inds = new int[nnz];
    block.values = new float[nnz];

    int i, p;
    for (i = 0, p = 0; p < nnz; i++, p += n) block.row_ptrs[i] = p;
    for (; i <= m; i ++) block.row_ptrs[i] = std::min(p, nnz);

    for (int i = 0; i < nnz; i++) block.col_inds[i] = i % n;
    for (int i = 0; i < nnz; i++) block.values[i] = (float) rand() / (float) (RAND_MAX);;

    return block;
}

struct CSRBlock construct_sample_csc_block(int m, int n, double nnz) {
    struct CSRBlock block;

    //int nnz = std::ceil((m*n) * sparsity);

    block.row_ptrs = new int[m + 1];
    block.col_inds = new int[nnz];
    block.values = new float[nnz];

    int i, p;
    for (i = 0, p = 0; p < nnz; i++, p += n) block.row_ptrs[i] = p;
    for (; i <= m; i ++) block.row_ptrs[i] = std::min(p, nnz);

    for (int i = 0; i < nnz; i++) block.col_inds[i] = i % n;
    for (int i = 0; i < nnz; i++) block.values[i] = (float) rand() / (float) (RAND_MAX);;

    return block;
}



void bench_4_4_4() {
    typedef Vec<float, 512>::Type VecType;
    typedef StorageTypes<float, int, int> StorageTypes;
    static constexpr int runs = 10000000;
    sym_lib::timing_measurement t1;

    static constexpr int m = 4;
    static constexpr int k = 4;

    static constexpr int B_VECS = 4;
    static constexpr int n = VecType::size() * B_VECS;

    auto coo_block = construct_sample_coo_block(m, k);

    float *B = new float[k * n]();
    float *C = new float[m * n]();

    for (double density = 0.05; density < 1; density += 0.05) {
        auto csr_block = construct_sample_csr_block(m, k, density);

        delete[] csr_block.values;
        csr_block.values = coo_block.values;

        std::cout << m << ", " << k << ", " << B_VECS << ", ";
        std::cout << density << ", ";

        std::fill(B, B + VecType::size() * m * B_VECS, 1);
        std::fill(C, C + VecType::size() * k * B_VECS, 0);

        for (int iter = 0; iter < 5000; iter++) {
            _a_dense_times_b_col_panel_4_4_1<VecType, StorageTypes>(
                    m, k, n,
                    0, 0,
                    coo_block.values, coo_block.row_inds, coo_block.col_inds,
                    B, C);
        }


        t1 = sym_lib::timing_measurement();
        t1.start_timer();
        for (int run = 0; run < runs; run++) {
            _a_dense_times_b_col_panel_4_4_1<VecType, StorageTypes>(
                    m, k, n,
                    0, 0,
                    coo_block.values, coo_block.row_inds, coo_block.col_inds,
                    B, C
            );
        }

        t1.measure_elapsed_time();
        std::cout << t1.elapsed_time / runs << ", ";

        for (int iter = 0; iter < 5000; iter++) {
            _a_row_times_b_col_panel_1<VecType, StorageTypes, false, true>(
                    m, k, n,
                    0, 0,
                    nullptr,
                    csr_block.values, csr_block.row_ptrs, csr_block.col_inds,
                    B, C);
        }

        t1 = sym_lib::timing_measurement();
        t1.start_timer();
        for (int run = 0; run < runs; run++) {
            _a_row_times_b_col_panel_1<VecType, StorageTypes, false, true>(
                    m, k, n,
                    0, 0,
                    nullptr,
                    csr_block.values, csr_block.row_ptrs, csr_block.col_inds,
                    B, C
            );
        }
        t1.measure_elapsed_time();
        std::cout << t1.elapsed_time / runs << ", ";
        std::cout << std::endl;

        delete[] csr_block.row_ptrs;
        delete[] csr_block.col_inds;
    }

    delete[] coo_block.values;
    delete[] coo_block.col_inds;
    delete[] coo_block.row_inds;
    delete[] B;
    delete[] C;
}


int main(int argc, char *argv[]) {

    bench_4_4_4();

    return 0;
}