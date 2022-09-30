//
// Created by lwilkinson on 5/28/22.
//

#include "OverlapPctRowDistance.h"


OverlapPctRowDistance::OverlapPctRowDistance(SparsityPattern &pattern):
    RowDistance(pattern.num_rows_, pattern.num_cols_) {
    use_bitset = uint64_t(pattern.num_rows_) * uint64_t(pattern.num_cols_) < 1e10;

    if (use_bitset) {
        build_row_bitsets(pattern.ptrs_, pattern.indices_);
    } else {
        row_ptrs = pattern.ptrs_;
        col_indices = pattern.indices_;
    }
}


void OverlapPctRowDistance::build_row_bitsets(int row_ptrs[], int col_indices[]) {
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

double OverlapPctRowDistance::dist(int row1, int row2) {
    if (use_bitset) {
        return (row_bitsets[row1] & row_bitsets[row2]).count() / (row_bitsets[row1] | row_bitsets[row2]).count();
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
        return double(overlap) / total_nnz;
    }
}

double OverlapPctRowDistance::panel_dist(int start_row, int end_row) {
    if (use_bitset) {
        auto row_union = row_bitsets[start_row];
        int total_nnz = row_bitsets[start_row].count();

        for (int i = start_row + 1; i < end_row; i++) {
            total_nnz += row_bitsets[i].count();
            row_union |= row_bitsets[i];
        }

        return double(total_nnz - row_union.count()) / total_nnz;
    } else {
        std::set<int> row_union;
        int total_nnz = 0;

        for (int i = start_row; i < end_row; i++) {
            total_nnz += row_ptrs[i + 1] - row_ptrs[i];
            for (int p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
                row_union.insert(col_indices[p]);
            }
        }

        return double(total_nnz - row_union.size()) / total_nnz;
    }
}