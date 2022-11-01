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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"
#include <omp.h>
#include <cstdlib>

#include "../SpMMFunctor.h"

using namespace arm_compute;
using namespace utils;


template<typename Scalar>
class SpMMARMCLDense : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;
    Scalar* a_dense;

    Tensor      A{}, B{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    size_t      M;
    size_t      N;
    size_t      K;
    int         p;

public:
    SpMMARMCLDense(typename Super::Task &task) : Super(task) {
        typename Super::Task& t = this->task;

        a_dense = new Scalar[t.m() * t.k()]();
        zero(a_dense, t.m() * t.k());

        for (int i = 0; i < t.A->r; i++) {
            for (int p = t.A->Lp[i]; p < t.A->Lp[i + 1]; p++) {
                a_dense[(i * t.k()) + t.A->Li[p]] = t.A->Lx[p];
            }
        }

        M = t.m();
        N = t.n();
        K = t.k();

        alpha = 1.0f;
        beta  = 0.0f;

        A.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        B.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));

        init_sgemm_output(dst, A, B, DataType::F32);

        // Allocate all the images
        A.allocator()->import_memory(a_dense);
        B.allocator()->import_memory(t.B);
        dst.allocator()->import_memory(t.C);

        // use nThreads cores for experiment
        CPPScheduler::get().set_num_threads(t.nThreads);
        NEScheduler::get().set_num_threads(t.nThreads);

        // Configure function
        sgemm.configure(&A, &B, nullptr, &dst, alpha, beta);

    }

    void copy_output() override {

    }

    ~SpMMARMCLDense() {
        delete[] a_dense;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;
        sgemm.run();
    }
};

