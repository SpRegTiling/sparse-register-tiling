//
// Created by lwilkinson on 6/10/22.
//

#pragma once

#include <math.h>
#include <vector>


#include "SpMMFunctor.h"
#include "spmm_config.h"
#include "utils/error.h"



struct XNNConfig: ConfigBase {

};


#define restrict __restrict


//#define MIN_MAX
//#define PRESCALE_DIFF

void xnn_f32_spmm_minmax_ukernel_16x1__neon(
        size_t mc,
        size_t nc,
        const float*restrict input,
        const float*restrict weights,
        const int32_t*restrict widx_dmap,
        const uint32_t*restrict nidx_nnzmap,
        float*restrict output,
        size_t output_stride
);

void xnn_f32_spmm_minmax_ukernel_16x1__neon_parallel(
        size_t input_size, // ncols
        size_t nc, // nrows
        const float*restrict input,
        const float*restrict weights,
        const int32_t*restrict widx_dmap,
        const uint32_t*restrict nidx_nnzmap,
        float*restrict output,
        size_t output_stride,
        size_t num_threads
);

template<typename Scalar>
class SpMM_XNN : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;

    XNNConfig config;

    float*    weights        = nullptr;
    int32_t*  col_diffs      = nullptr;
    int32_t*  col_increment  = nullptr;
    uint32_t* row_nnz        = nullptr;

    int32_t   first_nnz_diff = 0;


    void delete_storage() {
        free(weights);
        free(col_diffs);
        free(col_increment);
        free(row_nnz);
    }

public:

    void pack() {
        typename Super::Task& t = this->task;

        int32_t*  col_increment = (int32_t*) aligned_alloc(64, (t.A->nz + 1) * sizeof(int32_t));
        int32_t*  col_diffs = (int32_t*) aligned_alloc(64, (t.A->nz + 1) * sizeof(int32_t));
        float*    weights = (float*) aligned_alloc(64, (t.A->nz + t.m() + 1) * sizeof(int32_t));
        uint32_t* row_nnz = (uint32_t*) aligned_alloc(64, (t.m() + 1) * sizeof(uint32_t));

        this->col_increment = col_increment;
        this->col_diffs = col_diffs;
        this->weights = weights;
        this->row_nnz = row_nnz;

        size_t first_nnz_col = 0, last_nnz_col = 0;
        bool first_nonzero = true;

        for (int i = 0; i < t.m(); i++) {
            *weights++ = 0.0;
            *row_nnz = 0;

            for (int p = t.A->Lp[i]; p < t.A->Lp[i + 1]; p++) {
                if (first_nonzero) {
                    first_nnz_col = t.A->Li[p];
                } else {
                    const int64_t diff = (int64_t) ((uint64_t) t.A->Li[p] - (uint64_t) last_nnz_col) * sizeof(float);
                    *col_diffs++ = (int32_t) diff;
                }

                first_nonzero = false;
                (*row_nnz) += 1;
                *weights++ = t.A->Lx[p];
                last_nnz_col = (size_t) t.A->Li[p];
            }

            row_nnz += 1;
        }

        const int64_t diff = (int64_t) ((uint64_t) first_nnz_col - (uint64_t) last_nnz_col) * sizeof(float);
        *col_diffs++ = diff;
        first_nnz_diff = first_nnz_col;

        col_diffs = this->col_diffs;
        for (int i = 0; i <= t.A->nz; i++) {
#ifdef PRESCALE_DIFF
            *col_increment++ = (*col_diffs++) * t.n();
#else
            *col_increment++ = (*col_diffs++);
#endif
        }
    }

    SpMM_XNN(typename Super::Task &task) : Super(task) { pack(); }

    ~SpMM_XNN() { delete_storage(); }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;
    if(false) {
     xnn_f32_spmm_minmax_ukernel_16x1__neon(
       t.n() * sizeof(float),
       t.m(),
       t.B + first_nnz_diff * t.n(),
       weights,
       col_increment,
       row_nnz,
       t.C,
       t.n() * sizeof(float)
     );
    }else{
     xnn_f32_spmm_minmax_ukernel_16x1__neon_parallel(
       t.n() * sizeof(float),
       t.m(),
       t.B + first_nnz_diff * t.n(),
       weights,
       col_increment,
       row_nnz,
       t.C,
       t.n() * sizeof(float),
       t.nThreads
     );
     }
    }
};
