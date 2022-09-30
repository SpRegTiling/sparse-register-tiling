//
// Created by lwilkinson on 6/3/22.
//

#ifndef DNN_SPMM_BENCH_SPMMTASK_H
#define DNN_SPMM_BENCH_SPMMTASK_H

#include "Matrix.h"

template<typename Scalar>
struct SpMMTask {
    CSR<Scalar> *A;
    Scalar *B;
    Scalar *C;
    Scalar *correct_C;

    int bRows;
    int bCols;
    std::string filepath;
    int nThreads;

    int cRows() const { return A->r; }
    int cCols() const { return bCols; }
    int cNumel() const { return cRows() * cCols(); }

    int m() const { return A->r;  }
    int k() const { return A->c;  }
    int n() const { return bCols; }
};


#endif //DNN_SPMM_BENCH_SPMMTASK_H
