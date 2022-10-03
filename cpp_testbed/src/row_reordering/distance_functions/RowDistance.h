//
// Created by lwilkinson on 5/27/22.
//

#ifndef DNN_SPMM_BENCH_DISTANCE_H
#define DNN_SPMM_BENCH_DISTANCE_H

#include <string>


class RowDistance {
protected:
    int rows = 0;
    int cols = 0;

    RowDistance(int rows, int cols): rows(rows), cols(cols) {}

public:
    virtual ~RowDistance() = default;

    virtual std::string name() = 0;
    int num_rows() { return rows; }

    virtual double dist(int row1, int row2) = 0;
    virtual double panel_dist(int start_row, int end_row);
};

#endif //DNN_SPMM_BENCH_DISTANCE_H
