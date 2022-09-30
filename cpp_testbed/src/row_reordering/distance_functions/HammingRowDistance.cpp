//
// Created by lwilkinson on 5/27/22.
//

#include "HammingRowDistance.h"


HammingRowDistance::HammingRowDistance(SparsityPattern &pattern):
    RowDistance(pattern.num_rows_, pattern.num_cols_) {
    use_bitset = uint64_t(pattern.num_rows_) * uint64_t(pattern.num_cols_) < 1e10;

    if (use_bitset) {
        build_row_bitsets(pattern.ptrs_, pattern.indices_);
    } else {
        row_ptrs = pattern.ptrs_;
        col_indices = pattern.indices_;
    }
}

void HammingRowDistance::build_row_bitsets(int row_ptrs[], int col_indices[]) {
    row_bitsets = std::vector<boost::dynamic_bitset<>>(rows);
    for (auto& row_bitset : row_bitsets) {
        row_bitset.resize(cols);
    }

    for (int i = 0; i < rows; i++ ){
        for (int p = row_ptrs[i]; p < row_ptrs[i+1]; p ++) {
            row_bitsets[i].set(col_indices[p]);
        }
    }
}

double HammingRowDistance::dist(int row1, int row2) {
    if (use_bitset) {
        return (row_bitsets[row1] ^ row_bitsets[row2]).count();
    } else {
        int row1_start = row_ptrs[row1];
        int row1_p = row1_start;
        int row1_end = row_ptrs[row1 + 1];

        int row2_start = row_ptrs[row2];
        int row2_p = row2_start;
        int row2_end = row_ptrs[row2 + 1];

        int overlap = 0;

        while (row1_p < row1_end && row2_p < row2_end) {
            if (col_indices[row1_p] < col_indices[row2_p]) { row1_p++; continue; }
            if (col_indices[row2_p] < col_indices[row1_p]) { row2_p++; continue; }
            row1_p++; row2_p++; overlap++;
        }

        int total_nnz = (row1_end - row1_start) + (row2_end - row2_start);
        return total_nnz - overlap;
    }
}