//
// Created by lwilkinson on 6/4/22.
//

#include "row_reordering_algos.h"

#include <set>

std::vector<int> greedy_row_reordering(int panel_size, RowDistance& distance_measure) {
    std::set<int> visited;
    std::vector<int> row_swizzle(distance_measure.num_rows());

    visited.insert(0);
    row_swizzle[0] = 0;

    for (int i = 0; i < distance_measure.num_rows()-1; i++) {
        double shortest_distance = 1e9;
        int best_row = 0;

        for (int ii = 1; ii < distance_measure.num_rows(); ii++) {
            if (!visited.count(ii)) {
                auto dist = distance_measure.dist(row_swizzle[i], ii);
                if (dist < shortest_distance) {
                    best_row = ii;
                    shortest_distance = dist;
                }
            }
        }

        visited.insert(best_row);
        row_swizzle[i+1] = best_row;
    }

    return row_swizzle;
};