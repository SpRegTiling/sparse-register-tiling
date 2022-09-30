//
// Created by lwilkinson on 6/5/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_ASPT_H
#define DNN_SPMM_BENCH_SPMM_ASPT_H

#include "../SpMMFunctor.h"
#include "ASpT.h"

template<typename Scalar>
class SpMMASpT : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;

private:
    InspectorMetadata<Scalar> meta;

public:
    SpMMASpT(typename Super::Task &task) : Super(task) {
        meta = inspect(
            task.A->r, task.A->c, task.A->nz,
            task.A->Lp, task.A->Li, task.A->Lx,
            68
        );
    }

    ~SpMMASpT() {
        free(meta);
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;

        execute(meta,
            t.A->r, t.A->c, t.bCols,
            t.A->Lp, t.A->Li, t.A->Lx,
            t.B,
            t.C
        );
    }
};

#endif //DNN_SPMM_BENCH_SPMM_ASPT_H
