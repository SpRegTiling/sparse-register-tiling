//
// Created by lwilkinson on 6/1/22.
//

#ifndef DNN_SPMM_BENCH_ROW_REORDERING_ALGOS_H
#define DNN_SPMM_BENCH_ROW_REORDERING_ALGOS_H

#include <vector>
#include "RowDistance.h"

typedef std::vector<int> (row_reordering_algo_t)(int panel_size, RowDistance& distance_measure);

row_reordering_algo_t greedy_row_reordering;

#endif //DNN_SPMM_BENCH_ROW_REORDERING_ALGOS_H
