//
// Created by lwilkinson on 5/28/22.
//

#ifndef DNN_SPMM_BENCH_OVERLAPCOUNTROWDISTANCE_H
#define DNN_SPMM_BENCH_OVERLAPCOUNTROWDISTANCE_H

#include "Matrix.h"
#include "RowDistance.h"
#include "boost/dynamic_bitset.hpp"

class OverlapPctRowDistance: public RowDistance {
    std::vector<boost::dynamic_bitset<>> row_bitsets;
    int* row_ptrs = nullptr;
    int* col_indices = nullptr;

    bool use_bitset = false;
    void build_row_bitsets(int row_ptrs[], int col_indices[]);

public:
    OverlapPctRowDistance(SparsityPattern &pattern);
    double dist(int row1, int row2) override;
    double panel_dist(int start_row, int end_row) override;

    std::string name() override { return "overlap_pct"; }
};

#endif //DNN_SPMM_BENCH_OVERLAPCOUNTROWDISTANCE_H
