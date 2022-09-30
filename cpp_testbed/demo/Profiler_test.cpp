//
// Created by lwilkinson on 5/30/22.
//

#include <iostream>

#include "Profiler.h"

using namespace cpp_testbed;

class StridedAccessLoop {
public:
    static constexpr int WORKSPACE_SIZE = 1024 * 1024 * 8;

private:
    static volatile int array_to_access[WORKSPACE_SIZE];

    int stride = 1;

    static int read(volatile int *addr) {
        return *addr;
    }

public:
    StridedAccessLoop(int stride): stride(stride) {}

    void operator() () {
        for (int offset = 0; offset < stride; offset++) {
            for (int i = offset; i < WORKSPACE_SIZE; i += stride) {
                read(&array_to_access[i]);
            }
        }
    }
};

// Allocate Memory
volatile int StridedAccessLoop::array_to_access[StridedAccessLoop::WORKSPACE_SIZE];


int main() {
    StridedAccessLoop unit_stride(1);
    csv_row_t results;

    auto unit_stride_profiler = Profiler(PAPIWrapper::get_available_counter_codes(), &unit_stride);
    unit_stride_profiler.profile();
    unit_stride_profiler.log_counters(results);

    std::cout << "Unit Stride Results | ";
    std::cout << "WORKSPACE SIZE " << StridedAccessLoop::WORKSPACE_SIZE << std::endl;
    for (auto it = results.cbegin(); it != results.cend(); ++it)
        std::cout << it->first << ": " << it->second << std::endl;

    StridedAccessLoop non_unit_stride(64);
    auto non_unit_profiler = Profiler(PAPIWrapper::get_available_counter_codes(), &non_unit_stride);
    non_unit_profiler.profile();
    non_unit_profiler.log_counters(results);

    std::cout << "\n";
    std::cout << "Non Unit Stride Results | ";
    std::cout << "WORKSPACE SIZE " << StridedAccessLoop::WORKSPACE_SIZE << std::endl;
    for (auto it = results.cbegin(); it != results.cend(); ++it)
        std::cout << it->first << ": " << it->second << std::endl;
}