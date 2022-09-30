//
// Created by lwilkinson on 6/24/22.
//



#include <type_traits>
#include <map>
#include <cxxabi.h>
#include <filesystem>
#include <iostream>
#include <iomanip>      // std::setw
#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#include <omp.h>
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <malloc.h>

#include "Input.h"
#include "def.h"
#include "SparseMatrixIO.h"
#include "row_reordering_algos.h"
#include "csv_log_io.h"
#include "spmm.h" // From xformers/sparse/backends/cpu/spmm/
#include "spmm_utils.h"

#include "SpMMFunctor.h"
#include "SpMMGeneric.h"
#include "SpMMTask.h"
#include "SpMM_DCSB.h"
#include "SpMM_MKL.h"
#include "SpMM_SOP_Old.h"

#include "dense_kernels.hpp"
#include "utils/Vec.h"

#include <boost/preprocessor/repetition/repeat.hpp>

#include "experiment_mapping.h"

#ifdef MKL
#include <cmath>
#include <mkl.h>
#endif

#ifdef PAPI_AVAILABLE
#include "Profiler.h"
#endif

struct Shape {
    int rows;
    int cols;

    int area() const { return rows * cols; }
};

struct COOBlocks {
    float* values;
    int* row_inds;
    int* col_inds;
};

using namespace cpp_testbed;
using namespace std;


template<typename T>
void pop_front(std::vector<T>& vec) {
    assert(!vec.empty());
    vec.erase(vec.begin());
}



#define DEF_C(z, _n, x)     VecType cVec##_n;
#define ZERO_C(z, _n, x)    cVec##_n = VecType(0);
#define LOAD_C(z, _n, x)    cVec##_n.load(&C[(row * ldn) + j + _n * VecType::size()]);
#define DEF_B(z, _n, x)     VecType bVec##_n;
#define LOAD_B(z, _n, x)    bVec##_n.load(&B[(b_row * ldn) + j + _n * VecType::size()]);
#define FMA_BC(z, _n, x)    cVec##_n = mul_add(aVec, bVec##_n, cVec##_n);
#define STORE_C(z, _n, x)   cVec##_n.store(&C[(row * ldn) + j + _n * VecType::size()]);

#define SPARSE_TILE_TEMPLATE(TILE_SIZE_IN_VECTORS)                                                          \
template<typename VecType, typename StorageTypes, bool row_swizzle, bool load_c>                            \
__always_inline int _sparse_tile_##TILE_SIZE_IN_VECTORS (                                                   \
    int m, int k, int n, int ldn,                                                                                 \
    int i, int j,                                                                                           \
    const int* __restrict__ row_indices,                                                                    \
    const typename StorageTypes::Scalar* __restrict__ values,                                               \
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,                                        \
    const typename StorageTypes::Index* __restrict__ column_indices,                                        \
    const typename StorageTypes::Scalar* __restrict__ B,                                                    \
    typename StorageTypes::Scalar* __restrict__ C                                                           \
) {                                                                                                         \
    for (; i < m; i ++) {                                                                                   \
        int row = i;                                                                                        \
                                                                                                            \
        for (j = 0; j < n; j += VecType::size() * TILE_SIZE_IN_VECTORS) {                                                                                                                \
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
            BOOST_PP_REPEAT(TILE_SIZE_IN_VECTORS, STORE_C, );                                               \
        }                                                                                                        \
    }                                                                                                       \
    return j + TILE_SIZE_IN_VECTORS * VecType::size();                                                      \
}


SPARSE_TILE_TEMPLATE(1);
SPARSE_TILE_TEMPLATE(2);
SPARSE_TILE_TEMPLATE(4);
SPARSE_TILE_TEMPLATE(8);

#define COO_BLOCK_SHAPE_ENUM(m, k, n) s##m##x##k##x##n
#define COO_BLOCK_SHAPE_MAP_ENUM(m, k, n, extra_cond) \
    if (shape.rows == m && shape.cols == k && extra_cond) return COO_BLOCK_SHAPE_ENUM(m, k, n);
#define COO_BLOCK_SHAPE_DISPATCH(_m, _k, _n) \
    case s##_m##x##_k##x##_n :  { _coo_block_##_m##_##_k##_x_##_n <VecType, StorageTypes>( \
        b_vecs, m, k, n, _i, j, values, row_indices, column_indices, B, C); break; }

enum COOBlockExecutor {
    COO_BLOCK_SHAPE_ENUM(4,  4, 1),
    COO_BLOCK_SHAPE_ENUM(4,  4, 2),
    COO_BLOCK_SHAPE_ENUM(4,  4, 4),
    COO_BLOCK_SHAPE_ENUM(4,  8, 1),
    COO_BLOCK_SHAPE_ENUM(4,  8, 2),
    COO_BLOCK_SHAPE_ENUM(4,  8, 4),
    COO_BLOCK_SHAPE_ENUM(4, 10, 1),
    COO_BLOCK_SHAPE_ENUM(4, 10, 2),
    COO_BLOCK_SHAPE_ENUM(4, 10, 4),
    COO_BLOCK_SHAPE_ENUM(4, 16, 1),
    COO_BLOCK_SHAPE_ENUM(4, 16, 2),
    COO_BLOCK_SHAPE_ENUM(4, 16, 4),
    COO_BLOCK_SHAPE_ENUM(4, 32, 1),
    COO_BLOCK_SHAPE_ENUM(4, 32, 2),
    COO_BLOCK_SHAPE_ENUM(4, 32, 4),
    COO_BLOCK_SHAPE_ENUM(6,  6, 1),
    COO_BLOCK_SHAPE_ENUM(6,  6, 2),
    COO_BLOCK_SHAPE_ENUM(6,  8, 1),
    COO_BLOCK_SHAPE_ENUM(6,  8, 2),
    COO_BLOCK_SHAPE_ENUM(6, 10, 1),
    COO_BLOCK_SHAPE_ENUM(6, 10, 2),
    COO_BLOCK_SHAPE_ENUM(6, 10, 4),
    COO_BLOCK_SHAPE_ENUM(6, 16, 1),
    COO_BLOCK_SHAPE_ENUM(6, 16, 2),
    COO_BLOCK_SHAPE_ENUM(8,  8, 1),
    COO_BLOCK_SHAPE_ENUM(8,  8, 2),
};

enum COOBlockExecutor map_shape_to_executor(Shape shape, int b_vecs) {
    COO_BLOCK_SHAPE_MAP_ENUM(4,  4, 4, b_vecs % 4 == 0);
    COO_BLOCK_SHAPE_MAP_ENUM(4,  8, 4, b_vecs % 4 == 0);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 10, 4, b_vecs % 4 == 0);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 16, 4, b_vecs % 4 == 0);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 32, 4, b_vecs % 4 == 0);
    COO_BLOCK_SHAPE_MAP_ENUM(6, 10, 4, b_vecs % 4 == 0);

    COO_BLOCK_SHAPE_MAP_ENUM(4,  4, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(4,  8, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 10, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 16, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(4, 32, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(6,  6, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(6,  8, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(6, 10, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(6, 16, 2, true);
    COO_BLOCK_SHAPE_MAP_ENUM(8,  8, 2, true);

    std::cerr << "unsupported coob shape" << std::endl; exit(0);
}

template<typename VecType, typename StorageTypes>
__always_inline void _coo_block_dispatch(
        enum COOBlockExecutor executor,
        int b_vecs,
        int m, int k, int n, int _i, int j,
        const typename StorageTypes::Scalar *__restrict__ values,
        const typename StorageTypes::Index *__restrict__ row_indices,
        const typename StorageTypes::Index *__restrict__ column_indices,
        const typename StorageTypes::Scalar *__restrict__ B,
        typename StorageTypes::Scalar *__restrict__ C)
{
    switch(executor) {
        COO_BLOCK_SHAPE_DISPATCH(4,  4, 2);
        COO_BLOCK_SHAPE_DISPATCH(4,  8, 2);
        COO_BLOCK_SHAPE_DISPATCH(4, 10, 2);
        COO_BLOCK_SHAPE_DISPATCH(4, 16, 2);
        COO_BLOCK_SHAPE_DISPATCH(4, 32, 2);
        COO_BLOCK_SHAPE_DISPATCH(6,  6, 2);
        COO_BLOCK_SHAPE_DISPATCH(6,  8, 2);
        COO_BLOCK_SHAPE_DISPATCH(6, 10, 2);
        COO_BLOCK_SHAPE_DISPATCH(6, 16, 2);
        COO_BLOCK_SHAPE_DISPATCH(8,  8, 2);

        COO_BLOCK_SHAPE_DISPATCH(4,  4, 1);
        COO_BLOCK_SHAPE_DISPATCH(4,  8, 1);
        COO_BLOCK_SHAPE_DISPATCH(4, 10, 1);
        COO_BLOCK_SHAPE_DISPATCH(4, 16, 1);
        COO_BLOCK_SHAPE_DISPATCH(4, 32, 1);
        COO_BLOCK_SHAPE_DISPATCH(6,  6, 1);
        COO_BLOCK_SHAPE_DISPATCH(6,  8, 1);
        COO_BLOCK_SHAPE_DISPATCH(6, 10, 1);
        COO_BLOCK_SHAPE_DISPATCH(6, 16, 1);
        COO_BLOCK_SHAPE_DISPATCH(8,  8, 1);

        COO_BLOCK_SHAPE_DISPATCH(4,  4, 4);
        COO_BLOCK_SHAPE_DISPATCH(4,  8, 4);
        COO_BLOCK_SHAPE_DISPATCH(4, 10, 4);
        COO_BLOCK_SHAPE_DISPATCH(4, 16, 4);
        COO_BLOCK_SHAPE_DISPATCH(4, 32, 4);
        COO_BLOCK_SHAPE_DISPATCH(6, 10, 4);
    }
}

std::tuple<Shape, Shape, std::vector<int>, std::vector<COOBlocks>> readCOOBFile(std::string path) {
    std::ifstream file;
    file.open(path, std::ios_base::in);
    if (!file.is_open()) { std::cout << "File could not be found " << path << std::endl; exit(1); }

    int i;
    std::istringstream line;

    auto get_linestream = [&file]() { std::string line; std::getline(file, line); return std::istringstream(line); };
    auto advance_to_nextline = [&file]() { char next; while (file.get(next)) { if (next == '\n') break; } };

    line = get_linestream();

    int rows, cols;
    line >> rows;
    line >> cols;

    line = get_linestream();

    int coob_rows, coob_cols;
    line >> coob_rows;
    line >> coob_cols;

    line = get_linestream();
    int tile_rows, tile_cols;
    line >> tile_rows;
    line >> tile_cols;

    line = get_linestream();
    int num_tiles;
    line >> num_tiles;

    std::vector<int> tile_num_blocks(num_tiles);
    std::vector<COOBlocks> tile_blocks(num_tiles);

    for (int t = 0; t < num_tiles; t++) {
        line = get_linestream();
        int num_blocks;
        line >> num_blocks;

        tile_num_blocks[t] = num_blocks;
        tile_blocks[t].values = new float[num_blocks * coob_rows * coob_cols];
        tile_blocks[t].row_inds = new int[num_blocks * coob_rows];
        tile_blocks[t].col_inds = new int[num_blocks * coob_cols];

        auto *values = tile_blocks[t].values;
        auto *row_inds = tile_blocks[t].row_inds;
        auto *col_inds = tile_blocks[t].col_inds;

        for (int b = 0; b < num_blocks; b++) {
            line = get_linestream();
            for (int i = 0; i < coob_rows * coob_cols; i++) line >> values[b * coob_rows * coob_cols + i];
            line = get_linestream();
            for (int i = 0; i < coob_rows; i++) line >> row_inds[b * coob_rows + i];
            line = get_linestream();
            for (int i = 0; i < coob_cols; i++) line >> col_inds[b * coob_cols + i];
        }
    }

    return { { coob_rows, coob_cols }, { tile_rows, tile_cols }, tile_num_blocks, tile_blocks };
}

template<typename T>
bool is_within_tol(const T &x, const T &y, const T scale_tol=32) {
    //http://realtimecollisiondetection.net/blog/?p=89
    auto relTol = scale_tol * std::max(std::abs(x), std::abs(y)) * std::numeric_limits<T>::epsilon();
    auto absTol = scale_tol * std::numeric_limits<T>::epsilon();

    return (std::abs(x - y) < relTol || std::abs(x - y) < absTol);
}

bool verify(float* C, float* C_sol, int size) {
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (!is_within_tol<float>(C[i], C_sol[i])) {
            //std::cout << i << " " << C[i] << " " << C_sol[i] << std::endl;
            correct = false;
        }
    }
    return correct;
}


int main(int argc, char *argv[]) {
    typedef Vec<float, 512>::Type VecType;
    typedef StorageTypes<float, int, int> StorageTypes;
    typedef cpp_testbed::CSR<float> CSR;

    auto As = cpp_testbed::readSparseMatrix<CSR>("/sdb/datasets/synthetic/coo_block_tile_32x64_32x64_4x16_75.0.mtx");

    mkl_set_num_threads(1);
    mkl_set_num_threads_local(1);
    mkl_set_dynamic(0);
    omp_set_num_threads(1);

    std::string DATASET_DIR = "/sdb/datasets/";

    std::vector<std::pair<std::string, double>> file_list = {\
//        { DATASET_DIR + "/synthetic/coo_block_matrix_512x512_4x8_75.0",   0.75},
//        { DATASET_DIR + "/synthetic/coo_block_matrix_512x512_4x8_100.0",  1.0},
        { DATASET_DIR + "/synthetic/coo_block_matrix_512x512_32x64_4x16_75.0.mtx",   0.75},
        { DATASET_DIR + "dlmc/transformer/magnitude_pruning/0.7/body_decoder_layer_4_encdec_attention_multihead_attention_q_fully_connected.smtx",   0.75},
        { DATASET_DIR + "dlmc/transformer/magnitude_pruning/0.7/body_decoder_layer_3_encdec_attention_multihead_attention_q_fully_connected.smtx",   0.75},
        { DATASET_DIR + "dlmc/transformer/magnitude_pruning/0.7/body_decoder_layer_5_encdec_attention_multihead_attention_q_fully_connected.smtx",   0.75},
        { DATASET_DIR + "dlmc/transformer/magnitude_pruning/0.7/body_decoder_layer_2_encdec_attention_multihead_attention_q_fully_connected.smtx",   0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x8_100.0",  1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x10_75.0",  0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x10_85.0",  0.85},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x10_100.0", 1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x64_32x64_6x10_75.0",  0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x16_85.0",  0.85},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_4x16_100.0", 1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x6_75.0",   0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x6_100.0",  1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x8_75.0",   0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x8_100.0",  1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x10_75.0",  0.75},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_6x10_100.0", 1.0},
//        { DATASET_DIR + "/synthetic/coo_block_tile_32x32_8x8_75.0",   0.75},
//        { DATASET_DIR + "/synthetic/coo_block_matrix_64x64_8x16_75.0",   0.75},
    };

    int runs = 1000;
    int warm_up_runs = 500;
    int profiler_runs = 5;

    static constexpr int coob_n_tile = 128;

    for (const auto& [file, coo_block_density] : file_list) {
        //std::string coob_file_path = file + ".coob";
        std::string mtx_file_path = file; //+ ".mtx";
        sym_lib::timing_measurement t1;

        std::cout << file << std::endl;

        auto As = cpp_testbed::readSparseMatrix<CSR>(mtx_file_path);
//        auto [_coob_shape, _tile_shape, _num_blocks, _coo_blocks] = readCOOBFile(coob_file_path);
//        // Hack to avoid issues with structured bindings and lambda captures
//        auto& coob_shape = _coob_shape; auto& tile_shape = _tile_shape;
//        auto& num_blocks = _num_blocks; auto& coo_blocks = _coo_blocks;

//        int coob_num_i_tiles = As.r / tile_shape.rows;
//        int coob_num_j_tiles = As.c / tile_shape.cols;

//        for (int i = 0; i < As.r / 4; i++) {
//            std::cout << i *4 << ": ";
//            for (int pat_idx = 0; pat_idx < panel_descs[i].num_patterns; pat_idx++) {
//                std::cout << "(" << (int) panel_descs[i].pattern_descs[pat_idx].pattern() << ", " << (int) panel_descs[i].pattern_descs[pat_idx].count() << ") ";
//            }
//            std::cout << std::endl;
//        };

        for (int B_VECS : { 8, 16 }) {
            std::cout << B_VECS << " ";

            csv_row_t csv_row;

            csv_row_insert(csv_row, "rows", As.r);
            csv_row_insert(csv_row, "cols", As.c);
            csv_row_insert(csv_row, "nnz", As.nz);
            csv_row_insert(csv_row, "matrixPath", file);
            csv_row_insert(csv_row, "coo_block_density", coo_block_density);
//            csv_row_insert(csv_row, "coob_rows", coob_shape.rows);
//            csv_row_insert(csv_row, "coob_cols", coob_shape.cols);
//            csv_row_insert(csv_row, "num_tiles", num_blocks.size());
//            csv_row_insert(csv_row, "num_blocks_total",  std::accumulate(num_blocks.begin(), num_blocks.end(), 0));

            csv_row_insert(csv_row, "B_VECS", B_VECS);
            csv_row_insert(csv_row, "b cols", B_VECS * VecType::size());

//            auto executor = map_shape_to_executor(coob_shape, B_VECS);

            int m = As.r;
            int k = As.c;
            int n = B_VECS * VecType::size(); //std::max(B_VECS, 4) * VecType::size();
            int n_partial = B_VECS * VecType::size();

            auto a_dense = new float[m * k]();
            zero(a_dense, m * k);

            for (int i = 0; i < As.r; i++) {
                for (int p = As.Lp[i]; p < As.Lp[i + 1]; p++) {
                    a_dense[(i * k) + As.Li[p]] = As.Lx[p];
                }
            }

            float *B = new float[k * n];
            float *C = new float[m * n];
            float *C_sol = new float[m * n];

            SpMMTask<float> spmm_task;
            spmm_task.A = &As;
            spmm_task.B = B;
            spmm_task.C = C;
            spmm_task.correct_C = C_sol;
            spmm_task.bRows = As.c;
            spmm_task.bCols = B_VECS * VecType::size();

            for (int i = 0; i < k*n; i++) { B[i] = i; }


            auto dcsb_functor = SpMM_DCSB<float, 512, uint8_t, DCSB_CSR>(spmm_task);
            auto sop_functor = SpMM_SOP<float, 512>(spmm_task);

            auto dscb_execute = [&]() mutable { dcsb_functor(); };

            auto rsop_execute = [&]() mutable {
                sop_functor(); };

//            auto coo_execute = [&]() mutable {
//                zero(C, m * n);
//                int b_vecs_per_tile = n / VecType::size();
//
//                if (n >= 128 && n % coob_n_tile == 0) {
//                    b_vecs_per_tile = coob_n_tile / VecType::size();
//                }
//
//                for (int ti = 0; ti < coob_num_i_tiles; ti++) {
//                    for (int kk = 0; kk < n; kk += b_vecs_per_tile * VecType::size()) {
//                        for (int tj = 0; tj < coob_num_j_tiles; tj++) {
//                            int tid = ti * coob_num_j_tiles + tj;
//
//                            for (int b = 0; b < num_blocks[tid]; b++) {
//                                static_assert(coob_n_tile % VecType::size() == 0);
//                                _coo_block_dispatch<VecType, StorageTypes>(
//                                        executor,
//                                        b_vecs_per_tile,
//                                        m, k, n,
//                                        0, 0,
//                                        &coo_blocks[tid].values[b * coob_shape.area()],
//                                        &coo_blocks[tid].row_inds[b * coob_shape.rows],
//                                        &coo_blocks[tid].col_inds[b * coob_shape.cols],
//                                        &B[kk], &C[kk]
//                                );
//                            }
//                        }
//                    }
//                }
//            };

            auto csr_execute = [&]() mutable {
                zero(C, m * n);

                if (n_partial % (16 * 8) == 0) {
                    _sparse_tile_4<VecType, StorageTypes, false, true>(
                            m, k, n_partial, n,
                            0, 0,
                            nullptr,
                            As.Lx, As.Lp, As.Li,
                            B, C
                    );
                } else if (n_partial % (16 * 4) == 0) {
                    _sparse_tile_4<VecType, StorageTypes, false, true>(
                            m, k, n_partial, n,
                            0, 0,
                            nullptr,
                            As.Lx, As.Lp, As.Li,
                            B, C
                    );
                } else {
                    _sparse_tile_2<VecType, StorageTypes, false, true>(
                            m, k, n_partial, n,
                            0, 0,
                            nullptr,
                            As.Lx, As.Lp, As.Li,
                            B, C
                    );
                }
            };

            auto dense_execute = [&]() mutable {
                cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n_partial, k,
                        1.,                 // alpha
                        a_dense, k,         // lda = t.k()
                        B, n,               // ldb = t.n()
                        0.,                 // beta
                        C, n                // ldc = t.n()
                );
            };

            zero(C, m * n);
            dense_execute();
            std::copy(C, &C[m*n], C_sol);



            for (auto& [executor, name] : std::vector<std::pair<std::function<void()>, std::string>> {
                    { dscb_execute,  "dcsb"  },
                    { rsop_execute,  "rsop"  },
//                    { coo_execute,   "coob"  },
                    { csr_execute,   "csr"   },
                    { dense_execute, "dense" }
            }) {
                for (int run = 0; run < warm_up_runs; run++) { executor(); }

                t1 = sym_lib::timing_measurement();
                t1.start_timer();
                for (int run = 0; run < runs; run++) { executor(); }
                t1.measure_elapsed_time();

                csv_row_insert(csv_row, "name", name);
                csv_row_insert(csv_row, "time", t1.elapsed_time / runs);

                if (!verify(C, C_sol, m * n)) {
                    std::cout << std::endl;
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < n; j ++ ) {
                            std::cout << "(" << C[i*n + j] << " " << C_sol[i*n + j] << ") ";
                        }
                        std::cout << std::endl;
                    }
                }

                std::cout << name << " " << t1.elapsed_time / runs << " ";

                auto& _executor = executor;
                auto run = std::function<void()>([&_executor, profiler_runs]() {
                    for (int i = 0; i < profiler_runs; i++) { _executor(); }
                });

                auto profiler = Profiler<std::function<void()>>(&run);
                profiler.profile();
                profiler.log_counters(csv_row);

                write_csv_row("coob_results.csv", csv_row);
            }

            std::cout << std::endl;

            delete[] C;
            delete[] C_sol;
            delete[] B;
        }

//        for (const auto& tile_coob_blocks : coo_blocks) {
//            delete[] tile_coob_blocks.values;
//            delete[] tile_coob_blocks.row_inds;
//            delete[] tile_coob_blocks.col_inds;
//        }
    }
}