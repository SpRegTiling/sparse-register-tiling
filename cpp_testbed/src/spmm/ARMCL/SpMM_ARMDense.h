//
// Created by lwilkinson on 6/8/22.
//

#ifndef DNN_SPMM_BENCH_DENSE_H
#define DNN_SPMM_BENCH_DENSE_H

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

using namespace arm_compute;
using namespace utils;

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha = 1.0f;
        beta  = 0.0f;

        M = strtol(argv[1], nullptr, 10);
        N = strtol(argv[2], nullptr, 10);
        K = strtol(argv[3], nullptr, 10);
        p = atoi(argv[4]);

        printf("M = %ld, K = %ld, N = %ld, p = %d\n", M,K,N,p);

        src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
        src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));


        init_sgemm_output(dst, src0, src1, DataType::F32);

        // Configure function
        sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();

        src2.allocator()->allocate();

        fill_random_tensor(src0, -1.f, 1.f);
        fill_random_tensor(src1, -1.f, 1.f);
        fill_random_tensor(src2, -1.f, 1.f);

        // Dummy run for CLTuner
        struct timespec start, end;
        double diff_t;

        clock_gettime(CLOCK_REALTIME, &start);

        sgemm.run();

        clock_gettime(CLOCK_REALTIME, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
        printf("sgemm 1 time: %f \n", diff_t);


        return true;
    }

    void do_run() override
    {
        // Execute the function
        struct timespec start, end;
        double diff_t;

        clock_gettime(CLOCK_REALTIME, &start);

        // use p cores for experiment
        NEScheduler::get().set_num_threads(p);
        sgemm.run();

        clock_gettime(CLOCK_REALTIME, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
        printf("sgemm time: %f \n", diff_t);

        char fname[50];
        snprintf(fname, sizeof(fname), "results_sq");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "armcl,%d,%ld,%f\n",p,M,diff_t);
        fclose(fp);

    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    size_t      M;
    size_t      N;
    size_t      K;
    int         p;
    bool        is_fortran{};
    std::string output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}
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
