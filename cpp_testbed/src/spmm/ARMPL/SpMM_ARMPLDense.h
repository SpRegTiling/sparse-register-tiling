//
// Created by lwilkinson on 6/8/22.
//

#pragma once

/*
 * Copyright (c) 2018-2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "armpl.h"

#include <omp.h>
#include <cstdlib>

#include "../SpMMFunctor.h"

#define ARMPL_CBLAS_TYPED_DISPATCH(func, ...)          \
    if constexpr(std::is_same_v<Scalar, float>)      \
        cblas_s##func (__VA_ARGS__);                 \
    else                                             \
        cblas_d##func (__VA_ARGS__);



template<typename Scalar>
class SpMMARMPLDense : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;
    Scalar* a_dense;

public:
    SpMMARMPLDense(typename Super::Task &task) : Super(task) {
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

    ~SpMMARMPLDense() {
        delete[] a_dense;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;

        // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
        ARMPL_CBLAS_TYPED_DISPATCH(gemm,
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
