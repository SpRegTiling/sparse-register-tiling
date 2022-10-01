//
// Created by lwilkinson on 9/30/22.
//

#ifndef NANO_SPMM_BENCH_ROW_REODERING_RUNTIME_MAPPING_H
#define NANO_SPMM_BENCH_ROW_REODERING_RUNTIME_MAPPING_H

#include <map>
#include <functional>

#include "Matrix.h"
#include "RowDistance.h"
#include "row_reordering_algos.h"

// Distance Mapping
using distance_factory_t = std::function<RowDistance* (SparsityPattern& pattern)>;
extern std::map<std::string, distance_factory_t> distance_mapping;

// Row-Reordering Algo Mapping
extern std::map<std::string, row_reordering_algo_t*> algo_mapping;

#endif //NANO_SPMM_BENCH_ROW_REODERING_RUNTIME_MAPPING_H
