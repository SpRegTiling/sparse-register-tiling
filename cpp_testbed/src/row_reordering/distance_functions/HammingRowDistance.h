//
// Created by lwilkinson on 5/27/22.
//

#ifndef DNN_SPMM_BENCH_HAMMING_H
#define DNN_SPMM_BENCH_HAMMING_H

#include "Matrix.h"
#include "RowDistance.h"
#include "boost/dynamic_bitset.hpp"

class HammingRowDistance: public RowDistance {
    std::vector<boost::dynamic_bitset<>> row_bitsets;
    int* row_ptrs = nullptr;
    int* col_indices = nullptr;

    bool use_bitset = false;
    void build_row_bitsets(int row_ptrs[], int col_indices[]);

public:
    HammingRowDistance(SparsityPattern &pattern);
    double dist(int row1, int row2) override;
    std::string name() { return "hamming"; }
};


#endif //DNN_SPMM_BENCH_HAMMING_H
