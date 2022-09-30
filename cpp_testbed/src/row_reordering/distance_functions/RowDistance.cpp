//
// Created by lwilkinson on 5/28/22.
//

#include "RowDistance.h"


double RowDistance::panel_dist(int start_row, int end_row) {
    int panel_distance = 0;
    auto end = (std::min(end_row, rows) - 1);

    for (int i = start_row; i < end; i ++) {
        auto dist = this->dist(i, i + 1);
        panel_distance += dist;
    }

    return panel_distance;
}