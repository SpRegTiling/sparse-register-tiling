//
// Created by Faraz on 10/6/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_MKL_BSR_H
#define DNN_SPMM_BENCH_SPMM_MKL_BSR_H

#include "../SpMMFunctor.h"
#include "MKL_utils.h"
#include <iostream>

template<typename Scalar>
class SpMMMKLBSR : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;
    using IndexType = typeof(Super::Task::A->Lp);

private:
    matrix_descr d;
    sparse_matrix_t m;
    MKL_INT *mkl_row_ptrs;

public:
    SpMMMKLBSR(typename Super::Task &task, int block_size) : Super(task) {
        d.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_status_t status;
        sparse_matrix_t m_csr;

        if constexpr(std::is_same_v<IndexType, MKL_INT>) {
            mkl_row_ptrs = task.A->Lp;
        } else {
            mkl_row_ptrs = new MKL_INT[task.A->r + 1]();
            for (int l = 0; l < task.A->r + 1; ++l) mkl_row_ptrs[l] = task.A->Lp[l];
        }

        MKL_SPARSE_TYPED_DISPATCH(create_csr,
                                  &m_csr,
                                  SPARSE_INDEX_BASE_ZERO,
                                  task.A->r, task.A->c,
                                  &mkl_row_ptrs[0], &mkl_row_ptrs[1],
                                  task.A->Li,
                                  task.A->Lx);
        MKL_CHECK(status);

        mkl_sparse_convert_bsr(m_csr, block_size, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &m);
        MKL_CHECK(status);

        // if (run_inspector) {
        //     status = mkl_sparse_set_mm_hint(m, SPARSE_OPERATION_NON_TRANSPOSE,
        //                                     d, SPARSE_LAYOUT_ROW_MAJOR,
        //                                     task.bCols, 1e9);
        //     MKL_CHECK(status);

        //     status = mkl_sparse_set_memory_hint(m, SPARSE_MEMORY_AGGRESSIVE);
        //     MKL_CHECK(status);

        //     status = mkl_sparse_optimize(m);
        //     MKL_CHECK(status);
        // }
    }

    ~SpMMMKLBSR() {
        typename Super::Task& t = this->task;

        if constexpr(!std::is_same_v<IndexType, MKL_INT>) {
            delete[] mkl_row_ptrs;
        }

        sparse_status_t status = mkl_sparse_destroy(m);
        MKL_CHECK(status);
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() override {
        typename Super::Task& t = this->task;

        sparse_status_t status;
        MKL_SPARSE_TYPED_DISPATCH(mm,
                                  SPARSE_OPERATION_NON_TRANSPOSE, 1,
                                  m, d,
                                  SPARSE_LAYOUT_ROW_MAJOR,
                                  t.B, t.bCols, t.bCols,
                                  0,
                                  t.C, t.bCols);
        MKL_CHECK(status);
    }
};


#endif //DNN_SPMM_BENCH_SPMM_MKL_BSR_H
