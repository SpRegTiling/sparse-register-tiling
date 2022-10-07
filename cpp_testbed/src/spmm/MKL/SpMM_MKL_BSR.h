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
    int bsr_num_blocks, bsr_required_storage, csr_required_storage;

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

        int csr_nnz = 0;
        for(int i = 0; i < task.A->r; i++) {
            csr_nnz += task.A->Lp[i+1] - task.A->Lp[i];
        }
        csr_required_storage = 2 * csr_nnz + task.A->r + 1;

        mkl_sparse_convert_bsr(m_csr, block_size, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &m);
        MKL_CHECK(status);

        sparse_index_base_t m_indexing;
        sparse_layout_t m_block_layout;
        int m_rows, m_cols, m_block_size;
        int *m_rows_start, *m_rows_end, *m_col_indx;
        Scalar* m_values;
        
        MKL_SPARSE_TYPED_DISPATCH(export_bsr,
                                  m,
                                  &m_indexing,
                                  &m_block_layout, &m_rows, &m_cols,
                                  &m_block_size, &m_rows_start, &m_rows_end,
                                  &m_col_indx,
                                  &m_values);

        // mkl_sparse_s_export_bsr(m, indexing, block_layout, rows, cols, block_size, rows_start, rows_end, col_indx, values);
        MKL_CHECK(status);

        bsr_num_blocks = 0;
        bsr_required_storage = 0;
        for (int i = 0; i < m_rows; i++) {
            bsr_num_blocks += m_rows_end[i] - m_rows_start[i];
        }
        bsr_required_storage = bsr_num_blocks * m_block_size * m_block_size * sizeof(Scalar) + 2 * m_rows * sizeof(int) + bsr_num_blocks * sizeof(int);

    }

    void log_extra_info(cpp_testbed::csv_row_t& row) override {
        csv_row_insert(row, "mkl_csr_required_storage", csr_required_storage);
        csv_row_insert(row, "mkl_bsr_num_blocks", bsr_num_blocks);
        csv_row_insert(row, "required_storage", bsr_required_storage);
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
