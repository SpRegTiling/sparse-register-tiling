//
// Created by lwilkinson on 6/1/22.
//

#ifndef DNN_SPMM_BENCH_GREEDYROWREORDERING_H
#define DNN_SPMM_BENCH_GREEDYROWREORDERING_H

#include "RowReordering.h"
#include "HammingRowDistance.h"
#include "custom_assert.h"

#include <set>

template<typename Distance>
class GreedyRowReordering: public RowReordering {
public:
    std::string name() override {
        return "greedy_" + std::string(typeid(Distance).name());
    }

    std::vector<int> operator() (SparsityPattern& pattern) override  {
        ASSERT_RELEASE(pattern.layout_ == SparsityPattern::CSR);
        auto dist_measure = Distance(pattern);

        std::set<int> visited;
        std::vector<int> row_swizzle(pattern.num_rows_);

//        for (int i = 0; )

        visited.insert(0);
        row_swizzle[0] = 0;

        for (int i = 0; i < pattern.num_rows_-1; i++) {
            double shortest_distance = 1e9;
            int best_row = 0;

            for (int ii = 1; ii < pattern.num_rows_; ii++) {
                if (!visited.count(ii)) {
                    auto dist = dist_measure.dist(row_swizzle[i], ii);
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
};


#endif //DNN_SPMM_BENCH_GREEDYROWREORDERING_H
