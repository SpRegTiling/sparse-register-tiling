//
// Created by lwilkinson on 9/30/22.
//

#include "row_reordering_runtime_mapping.h"

#include "HammingRowDistance.h"

std::map<std::string, distance_factory_t> distance_mapping = {
        {"hamming", [](SparsityPattern& pattern) { return new HammingRowDistance(pattern); }}
};

std::map<std::string, row_reordering_algo_t*> algo_mapping = {
        {"greedy", greedy_row_reordering}
};
