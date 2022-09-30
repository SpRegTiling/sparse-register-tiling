//
// Created by lwilkinson on 6/3/22.
//

#ifndef DNN_SPMM_BENCH_SPMMFUNCTOR_H
#define DNN_SPMM_BENCH_SPMMFUNCTOR_H

#include <string>
#include <cxxabi.h>
#include <set>
#include <functional>

#include "utils/misc.h"

#include "SpMMTask.h"
#include "utils.h"
#include "csv_log_io.h"

template<typename Scalar>
class SpMMFunctor {
public:
    virtual void operator()() = 0;
    virtual ~SpMMFunctor() {}

    using ParameterGrid = std::map<std::string, std::vector<int>>;
    using Config = std::map<std::string, int>;
    using Task = SpMMTask<Scalar>;

    std::vector<std::tuple<Config, double, double>> saved_results;

protected:
    SpMMTask<Scalar> &task;

    Config latest_config;
    Config tuned_config;

    bool tuned = false;

    virtual bool set_config_impl(const Config& config) {
        std::cerr << "Setting config not supported by this method" << std::endl;
        return false;
    }

    virtual std::string get_config_rep_impl() {
        return "";
    }

public:
    SpMMFunctor(Task &task): task(task) {}

    virtual void log_extra_info(cpp_testbed::csv_row_t& row) { }

    constexpr static int TUNING_ITERATIONS = 3;
    int tuning_runs_per_iteration() {
      int runs_per_iteration = std::max((int) std::ceil(15e7 / (task.A->nz * task.bCols)), 1);
      runs_per_iteration *= std::max(1, task.nThreads);
      return runs_per_iteration;
    }

    virtual void tune(const ParameterGrid& grid, bool save_results = false) {
        Config current_config;
        int runs_per_iter = tuning_runs_per_iteration();
        double best_time = std::numeric_limits<double>::max();
        std::set<std::string> keys;

        saved_results = {};

        // Compute keys to tune over
        for (const auto& [param_name, search_values] : grid) {

            for (const auto& value : search_values) {
                if (param_name == "m_tile" && value > task.m()) continue;
                if (param_name == "n_tile" && value > task.n()) continue;
                if (param_name == "k_tile" && value > task.k()) continue;

                current_config[param_name] = value;
            }

            keys.insert(param_name);
        }

        // We can handle a n-dimensional grid by recursively calling _tune_over on one dimension at a time
        std::function<void(std::set<std::string>, bool)> _tune_over = \
        [&](std::set<std::string> tune_over, bool parallelize) {
            auto param = *tune_over.begin();
            //#pragma omp parallel for if(parallelize)
            for (const int value : grid.find(param)->second) {
                if (param == "m_tile" && value > task.m()) continue;
                if (param == "n_tile" && value > task.n()) continue;
                if (param == "k_tile" && value > task.k()) continue;

                current_config[param] = value;
                auto remaining_params = tune_over;
                remaining_params.erase(param);

                // If there are more parameters to tune over recurse
                if (remaining_params.size()) {
                    _tune_over(remaining_params, false);
                    continue;
                }

                if (!this->set_config(current_config))
                    continue;

                // Warm up run
                for (int run = 0; run < runs_per_iter; run++) (*this)();
                std::vector<double> timings(TUNING_ITERATIONS);

                for (int iter = 0; iter < TUNING_ITERATIONS; iter++) {
                    sym_lib::timing_measurement t1;

                    t1.start_timer();
                    for (int run = 0; run < runs_per_iter; run++) (*this)();
                    t1.measure_elapsed_time();

                    timings[iter] = t1.elapsed_time / runs_per_iter;
                }

                //#pragma omp critical
                {
                    if (save_results) {
                        saved_results.push_back({
                            current_config,
                            median(timings),
                            mean(timings)
                        });
                    }

                    double median_time = median(timings);
                    if (median_time < best_time) {
                        best_time = median_time;
                        this->tuned_config = current_config;
                    }
                }
            }
        };

        // For now do not parallelize for best accuracy
        _tune_over(keys, false);

        this->set_config(tuned_config);
        tuned = true;
    }

    Config get_tuned_config() {
        if (!tuned) std::cerr << "Fetching untuned config, may not match the actual config used" << std::endl;
        return tuned_config;
    }

    std::string get_config_rep() {
        return get_config_rep_impl();
    }

    bool set_config(const Config& config) {
        bool success = set_config_impl(config);
        if (success) latest_config = config;
        return success;
    }

    virtual void set_row_reordering(std::vector<int>& row_ordering) {
        int status; std::string class_name(abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status));
        std::cerr << class_name << " does not support row-reordring" << std::endl;
    }
};

#endif //DNN_SPMM_BENCH_SPMMFUNCTOR_H
