//
// Created by lwilkinson on 6/5/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_ASPT_H
#define DNN_SPMM_BENCH_SPMM_ASPT_H

#include "../SpMMFunctor.h"
#include "ASpT.h"
#include "utils/error.h"

template<typename Scalar>
class SpMMASpT : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;

private:
    InspectorMetadata<Scalar> meta;
    int block_height = 128;

public:
    SpMMASpT(typename Super::Task &task, int block_height) :
        Super(task), block_height(block_height) {
        if (block_height < 0) {
            this->block_height = (task.A->r + task.nThreads - 1) / task.nThreads;
            this->block_height =  pow(2, ceil(log(this->block_height)/log(2)));
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            meta = inspect(
                task.A->r, task.A->c, task.A->nz,
                task.A->Lp, task.A->Li, task.A->Lx,
                task.nThreads,
                this->block_height
            );
        } else {
            ERROR_AND_EXIT("ASpT does not support double");
        }
    }

    ~SpMMASpT() {
        if constexpr (std::is_same_v<Scalar, float>) {
            free(meta);
        }
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;

        if constexpr (std::is_same_v<Scalar, float>) {
            execute(meta,
                t.A->r, t.A->c, t.bCols,
                t.A->Lp, t.A->Li, t.A->Lx,
                t.B,
                t.C,
                block_height
            );
        } else {
            ERROR_AND_EXIT("ASpT does not support double");
        }
    }
};

#endif //DNN_SPMM_BENCH_SPMM_ASPT_H
