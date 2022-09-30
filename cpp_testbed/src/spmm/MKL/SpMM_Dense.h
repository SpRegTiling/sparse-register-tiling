//
// Created by lwilkinson on 6/8/22.
//

#ifndef DNN_SPMM_BENCH_DENSE_H
#define DNN_SPMM_BENCH_DENSE_H

#include <mkl.h>

#include "../SpMMFunctor.h"

#define MKL_CBLAS_TYPED_DISPATCH(func, ...)          \
    if constexpr(std::is_same_v<Scalar, float>)      \
        cblas_s##func (__VA_ARGS__);                 \
    else                                             \
        cblas_d##func (__VA_ARGS__);



template<typename Scalar>
class SpMMMKLDense : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;
    Scalar* a_dense;

public:
    SpMMMKLDense(typename Super::Task &task) : Super(task) {
        typename Super::Task& t = this->task;

        a_dense = new Scalar[t.m() * t.k()]();
        zero(a_dense, t.m() * t.k());

//        sgemm_pack_get_size(CblasRowMajor, CblasNoTrans, CblasNoTrans, t.m(), t.k(), t.k(), &t.A->nz, &t.A->nz);

        for (int i = 0; i < t.A->r; i++) {
            for (int p = t.A->Lp[i]; p < t.A->Lp[i + 1]; p++) {
                a_dense[(i * t.k()) + t.A->Li[p]] = t.A->Lx[p];
            }
        }
    }

    ~SpMMMKLDense() {
        delete[] a_dense;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;

        // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
        MKL_CBLAS_TYPED_DISPATCH(gemm,
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            t.m(), t.n(), t.k(),
            1.,              // alpha
            a_dense, t.k(),  // lda = t.k()
            t.B,     t.n(),  // ldb = t.n()
            0.,              // beta
            t.C,     t.n()   // ldc = t.n()
        );
    }
};

#endif //DNN_SPMM_BENCH_DENSE_H
