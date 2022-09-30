//
// Created by lwilkinson on 6/3/22.
//

#ifndef DNN_SPMM_BENCH_SPMMGENERIC_H
#define DNN_SPMM_BENCH_SPMMGENERIC_H

#include "../SpMMFunctor.h"
#include "spmm.h"

template<typename StorageTypes, typename Config>
class SpMMGeneric : public SpMMFunctor<typename StorageTypes::Scalar> {
private:
    typedef typename StorageTypes::Index Index;
    typedef typename StorageTypes::IndexPtr IndexPtr;
    typedef typename StorageTypes::Scalar Scalar;

    using Super = SpMMFunctor<Scalar>;

    IndexPtr* row_ptrs = nullptr;
    Index*    col_inds = nullptr;

    bspmm_t<StorageTypes, Config>  *spmm;
    Config                          config;
    std::vector<int>                row_swizzle;

public:
    SpMMGeneric(
        SpMMTask<Scalar> &task,
        bspmm_t<StorageTypes, Config> *spmm
    ) : Super(task), spmm(spmm) {
        if constexpr(!std::is_same_v<Index, int>) {
            col_inds = new Index[task.A->nz];
            for (int i = 0; i < task.A->nz; i++) col_inds[i] = task.A->Li[i];
        } else {
            col_inds = task.A->Li;
        }

        if constexpr(!std::is_same_v<IndexPtr, int>) {
            row_ptrs = new IndexPtr[task.A->r + 1];
            for (int i = 0; i < (task.A->r + 1); i++) row_ptrs[i] = task.A->Lp[i];
        } else {
            row_ptrs = task.A->Lp;
        }
    }

    ~SpMMGeneric() {
        if constexpr(!std::is_same_v<Index, int>)     delete[] col_inds;
        if constexpr(!std::is_same_v<IndexPtr , int>) delete[] row_ptrs;
    }

    void set_row_reordering(std::vector<int>& row_ordering) override {
        row_swizzle = row_ordering;
    }

    bool set_config_impl(const typename Super::Config& config) override {
        for (const auto& [param_name, value] : config)
            this->config.setVal(param_name, value);

        return true;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() override {
        typename Super::Task& t = this->task;

        spmm(
            t.A->r, t.bRows, t.bCols,            // m, k, n
            t.A->nz,                             // nnz
            (row_swizzle.size()) ? row_swizzle.data() : nullptr,  // row_indices
            t.A->Lx,                             // values
            row_ptrs,                            // row_offsets
            col_inds,                            // column_indices
            t.B,                                 // dense_matrix
            t.C,                                 // out
            1,                                   // batch_size
            config
        );
    }
};

#endif //DNN_SPMM_BENCH_SPMMGENERIC_H
