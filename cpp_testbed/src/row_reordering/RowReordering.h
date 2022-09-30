//
// Created by lwilkinson on 6/1/22.
//

#ifndef DNN_SPMM_BENCH_ROWREORDERING_H
#define DNN_SPMM_BENCH_ROWREORDERING_H

#include <vector>

#include "Matrix.h"

class RowReordering {
public:
    virtual std::vector<int> operator() (SparsityPattern& pattern) = 0;
    virtual std::string name() = 0;
};


#endif //DNN_SPMM_BENCH_ROWREORDERING_H
