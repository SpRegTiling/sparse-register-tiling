//
// Created by lwilkinson on 6/3/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_TACO_H
#define DNN_SPMM_BENCH_SPMM_TACO_H

#include "../SpMMFunctor.h"
#include "utils/error.h"

#include <iostream>

#define restrict __restrict__
#include "taco.h"

extern int compute_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B);
extern int compute_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B);


template<typename Scalar>
class SpMMTACO : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;
    using IndexType = typeof(Super::Task::A->Lp);

private:
    taco_tensor_t* A;
    taco_tensor_t* B;
    taco_tensor_t* C;
    int width = 4;

public:
    SpMMTACO(typename Super::Task &task, int width = 4) : Super(task), width(width) {
        int A_dims[2] = {task.m(), task.k()};
        int B_dims[2] = {task.k(), task.n()};
        int C_dims[2] = {task.m(), task.n()};

        int mode_order[2] = {0, 1};
        taco_mode_t A_mode_types[2] = {taco_mode_dense, taco_mode_sparse};
        taco_mode_t B_mode_types[2] = {taco_mode_dense, taco_mode_dense};
        taco_mode_t C_mode_types[2] = {taco_mode_dense, taco_mode_dense};

        A = init_taco_tensor_t(2, 0, A_dims, mode_order, A_mode_types);
        B = init_taco_tensor_t(2, 0, B_dims, mode_order, B_mode_types);
        C = init_taco_tensor_t(2, 0, C_dims, mode_order, C_mode_types);

        A->vals = (uint8_t*) task.A->Lx;
        A->indices[1][0] = (uint8_t*) task.A->Lp;
        A->indices[1][1] = (uint8_t*) task.A->Li;

        B->vals = (uint8_t*) task.B;
        C->vals = (uint8_t*) task.C;
    }

    ~SpMMTACO() {
        free(A);
        free(B);
        free(C);
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() override {
        switch(width) {
            case 4: compute_4(C, A, B);
            case 16: compute_16(C, A, B);
        }
    }

    void copy_output() override {
        typename Super::Task& t = this->task;
        memcpy(t.C, C->vals, t.m() * t.n() * sizeof(Scalar));
    }
};


#endif //DNN_SPMM_BENCH_SPMM_TACO_H
