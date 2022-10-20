//
// Created by cetinicz on 2021-10-30.
//

#include <type_traits>
#include <map>
#include <cxxabi.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>      // std::setw
#include <optional>
#include <utility>
//#include <aligned_new>

#include <omp.h>
#include <ryml_std.hpp>
#include <ryml.hpp>

#include "utils/misc.h"
#include "utils/error.h"

// Google Benchmark
#include "benchmark/benchmark.h"

#include "Input.h"
#include "SparseMatrixIO.h"
#include "csv_log_io.h"
#include "def.h"
#include "row_reordering_algos.h"

#include "spmm/runtime_yaml_config.h"
#include "spmm/SpMMTask.h"
#include "spmm/SpMMFunctor.h"

#include "row_reordering_runtime_mapping.h"

#ifdef MKL
#include <cmath>
#include <mkl.h>
#endif

#ifdef PAPI_AVAILABLE
#include "Profiler.h"
#endif

#ifdef VTUNE_AVAILABLE
#include "ittnotify.h"
#endif

#include "cake_block_dims.h"

using namespace cpp_testbed;

bool report_packing_time = false;

/***********************************************************
 *  Benchmark utils
 **********************************************************/

template<typename Scalar>
Scalar *random_matrix(int m, int n) {
    Scalar *mtx = new(std::align_val_t(4096)) Scalar[m * n];
    for (int i = 0; i < m * n; i++) {
        mtx[i] = (Scalar) rand() / (Scalar) (RAND_MAX);
    }

    return mtx;
}

template<typename T>
bool l1_vec_norm(const T *x, int numel) {
    T acculmulator = 0;
    for (int i = 0; i < numel; i++) { acculmulator += std::abs(x[i]); }
    return acculmulator;
}

template<typename T>
bool is_within_tol(const T &x, const T &y, const T scale_tol=64) {
    //http://realtimecollisiondetection.net/blog/?p=89
    auto relTol = scale_tol * std::max(std::abs(x), std::abs(y)) * std::numeric_limits<T>::epsilon();
    auto absTol = scale_tol * std::numeric_limits<T>::epsilon();

    return (std::abs(x - y) < relTol || std::abs(x - y) < absTol);
}

template<typename T>
bool is_within_tol_binary_pred(const T &x, const T &y) {
    return is_within_tol<T>(x, y);
}

template<typename T>
bool verify(SpMMTask<T> &task) {
    for (int i = 0; i < task.cNumel(); i++) {
        if (!is_within_tol<T>(task.correct_C[i], task.C[i])) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool fast_verify(SpMMTask<T> &task) {
    // Quickly verify using L1 norm check

    auto l1_norm_correct = l1_vec_norm(task.correct_C, task.cNumel());
    auto l1_norm_C = l1_vec_norm(task.C, task.cNumel());

    return is_within_tol<T>(l1_norm_correct, l1_norm_C, task.cNumel());
}

template<typename T>
void report_mismatches(SpMMTask<T> &task) {
    int MAX_MISMATCHES = 32;
    bool REPORT_ZEROS = true;
    int MAX_ZEROS_TO_REPORT = 32;

    int num_mismatch = 0;
    for (int i = 0; i < task.cNumel(); i++) {
        if (!is_within_tol<T>(task.correct_C[i], task.C[i])) {
            std::cout << "(" << i / task.cCols() << ", " << i % task.cCols() << "): ";
            std::cout << task.correct_C[i] << " " << task.C[i] << std::endl;
            if (++num_mismatch >= MAX_MISMATCHES) break;
        }
    }

    if (!REPORT_ZEROS) return;

    int zero_mismatches = 0;
    for (int i = 0; i < task.cNumel(); i++) {
        if (!is_within_tol<T>(task.correct_C[i], task.C[i]) &&
            is_within_tol<T>(0, task.C[i])) {

            if (!zero_mismatches) std::cout << "ZERO Mismatches: " << std::endl;
            std::cout << "(" << i / task.cCols() << ", " << i % task.cCols() << "): ";
            std::cout << task.correct_C[i] << " " << task.C[i] << std::endl;
            if (++zero_mismatches >= MAX_ZEROS_TO_REPORT) break;
        }
    }
}

/***********************************************************
 *  Benchmark utils
 **********************************************************/

class SavedRunsReporter: public benchmark::BenchmarkReporter {
public:
    std::vector<std::vector<Run>> latest_runs;
    virtual bool ReportContext(const Context& context) BENCHMARK_OVERRIDE {
        return true;
    };
    virtual void ReportRuns(const std::vector<Run>& report) BENCHMARK_OVERRIDE {
        latest_runs.push_back(report);
    };

};

/***********************************************************
 *  Experiment
 **********************************************************/

template<typename Scalar>
class SpMMExperiment {
    struct Method {
        std::string name;
        std::string method_id;
        method_factory_t<Scalar> methodFactory;
        std::string rowReordering;
        std::string tuningParameterGrid;
        std::optional<typename SpMMFunctor<Scalar>::Config> config;
    };

    struct RowReorderingDefinition {
        std::string name;
        std::string algo;
        std::string distance;
    };

    using ParameterGrid = typename SpMMFunctor<Scalar>::ParameterGrid;
    using Config = typename SpMMFunctor<Scalar>::Config;

    std::vector<std::string> matricesToTest;
    std::vector<additional_options_t> additionalOptions;
    std::vector<struct Method> methods;
    std::map<std::string, struct RowReorderingDefinition> rowReorderingDefinitions;
    std::map<std::string, ParameterGrid> tuningParameterGrids;
    std::string outputFile;
    std::vector<std::string> expand_config_parameters;

    std::vector<int> perMatrixBColsToTest;
    std::vector<int> globalBColsToTest = { 256 };
    std::vector<int> nThreadsToTest = { 1 };

    cake_cntx_t* cake_cntx;

    bool profile = false;
    bool save_tuning_results = false;

    RowDistance* construct_distance(std::string distance_name, SparsityPattern& pattern) {
        if (!distance_mapping.count(distance_name)) {
            std::cerr << "Distance type " << distance_name << " not supported" << std::endl;
            exit(-1);
        }

        return distance_mapping[distance_name](pattern);
    };

    std::vector<int> compute_ordering(std::string algo_name, int panel_size, RowDistance* distance) {
        if (!algo_mapping.count(algo_name)) {
            std::cerr << "Row reordering algo " << algo_name << " not supported" << std::endl;
            exit(-1);
        }

        return algo_mapping[algo_name](panel_size, *distance);
    }


    std::string to_string(ryml::csubstr csubstr) {
        std::string str;
        ryml::from_chars(csubstr, &str);
        return str;
    }

    void parse_filelist(std::string filelistPath,
                        const std::vector<std::string>& search_dirs,
                        bool parse_b_cols_from_filelist,
                        bool filelist_has_additional_options) {
        filelistPath = resolve_path(filelistPath, search_dirs);
        int start_additional_options = parse_b_cols_from_filelist ? 2 : 1;

        std::ifstream file(filelistPath);
        std::string line;
        std::string filelistDir = std::filesystem::path(filelistPath).parent_path().string() + "/";
        std::vector<std::string> filelist_additional_options_names;

        if (!file.is_open()) {
            std::cout << "Failed to open " << filelistPath << std::endl;
            exit(-1);
        }

        auto split_line = [](std::string line) -> std::vector<std::string> {
            std::stringstream test(line);
            std::string segment;
            std::vector<std::string> seglist;
            while(std::getline(test, segment, ',')) seglist.push_back(segment);
            return seglist;
        };

        std::vector<std::string> _search_dirs = search_dirs;
        _search_dirs.insert(_search_dirs.begin() + 1, filelistDir);

        // Parse header
        if (filelist_has_additional_options) {
            std::getline(file, line);
            auto seglist = split_line(line);
            for (auto iter = seglist.begin() + start_additional_options;
                 iter != seglist.end(); ++iter) {
                filelist_additional_options_names.push_back(*iter);
            }
        }


        while (std::getline(file, line)) {
            auto seglist = split_line(line);

            std::string matrixPath = resolve_path(seglist[0], _search_dirs);
            matricesToTest.push_back(matrixPath);

            if (parse_b_cols_from_filelist) {
                int b_cols = std::stoi(seglist[1]);
                perMatrixBColsToTest.push_back(b_cols);
            }

            if (filelist_has_additional_options) {
              additional_options_t additional_options;
              for (int i = start_additional_options; i < seglist.size(); ++i) {
                auto name = filelist_additional_options_names
                    [i - start_additional_options];
                additional_options[name] = seglist[i];
              }

              additionalOptions.push_back(additional_options);
            }
        }

        file.close();
    }

    void parse_matrix_paths(c4::yml::ConstNodeRef paths, const std::vector<std::string>& search_dirs) {
        for(c4::yml::ConstNodeRef path : paths.children()) {
            std::string matrixPath;
            path >> matrixPath;
            matricesToTest.push_back(resolve_path(matrixPath, search_dirs));
        }
    }

    void parse_parameter_grids(c4::yml::ConstNodeRef parameter_grids) {
        for(c4::yml::ConstNodeRef parameter_grid : parameter_grids.children()) {
            ParameterGrid grid;
            std::string name; parameter_grid["name"] >> name;

            for(c4::yml::ConstNodeRef parameter : parameter_grid.children()) {
                if (parameter.key() == "name") continue;
                parameter >> grid[to_string(parameter.key())];
            }

            tuningParameterGrids[name] = grid;
        }
    }

    struct RowReorderingDefinition parse_row_reordering_definition(
        c4::yml::ConstNodeRef row_reordering) {
        struct RowReorderingDefinition def;

        if (!row_reordering.has_child("distance")) {
            std::cerr << "Row-reordering definition missing distance" << std::endl;
            exit(-1);
        }

        if (!row_reordering.has_child("name")) {
            std::cerr << "Row-reordering definition missing name" << std::endl;
            exit(-1);
        }

        if (!row_reordering.has_child("algo")) {
            std::cerr << "Row-reordering definition missing algo" << std::endl;
            exit(-1);
        }

        row_reordering["distance"] >> def.distance;
        row_reordering["name"] >> def.name;
        row_reordering["algo"] >> def.algo;
        return def;
    }

    struct Method parse_method(c4::yml::ConstNodeRef method_config) {
        std::string row_reordering, method_id, name, tuning_grid;

        if (method_config.has_child("row_reordering")) method_config["row_reordering"] >> row_reordering;
        if (method_config.has_child("tune")) {
            auto tuning = method_config["tune"];
            if (tuning.has_children() && tuning.has_child("grid")) tuning["grid"] >> tuning_grid;
        }

        method_config["method_id"] >> method_id;
        method_config["name"] >> name;

        if (get_method_id_mapping<Scalar>().count(method_id) == 0) {
            std::cerr << "No method_id " << method_id << " registered" << std::endl;
            exit(-1);
        }

        c4::yml::ConstNodeRef no_options;
        c4::yml::ConstNodeRef options = no_options;
        if (method_config.has_child("options")) options = method_config["options"];

        std::optional<typename SpMMFunctor<Scalar>::Config> config;
        if (method_config.has_child("config")) {
            auto config_yaml = method_config["config"];

            for(c4::yml::ConstNodeRef parameter : config_yaml.children()) {
                if (!config) config = typename SpMMFunctor<Scalar>::Config();
                parameter >> config.value()[to_string(parameter.key())];
            }
        }


        auto factory = get_method_id_mapping<Scalar>()[method_id](options);
        return { name, method_id, factory, row_reordering, tuning_grid, config };
    }

    void expand_config(csv_row_t& csv_row, const Config& config) {
        for (const auto& parameter : expand_config_parameters) {
            if (config.find(parameter) != config.end()) {
                csv_row_insert(csv_row, parameter, config.at(parameter));
            } else {
                csv_row_insert(csv_row, parameter, "");
            }
        }
    }

    /***********************************************************
     * Benchmark
     * @tparam Scalar
     * @param config
     * @return
     */
    int run_sample(const std::string& matrixPath,
                   const additional_options_t& additional_options,
                   std::vector<int> bColsToTest) {
        std::map<std::string, std::vector<int>> rowOrderings;
        std::map<std::string, RowDistance*> rowDistanceMeasures;

        constexpr int WARMUP_ITERATIONS = 0;
        constexpr int MEASURED_ITERATIONS = 1; // NOTE: Each measured iteration will consist of multiple runs

        typedef CSR<Scalar> CSR;
        auto As = cpp_testbed::readSparseMatrix<CSR>(matrixPath);

//        if (As.r % 32 != 0 || As.c % 32 != 0) return 0;

        std::cout << matrixPath << " (" << As.r << "x" << As.c << ") ";
        std::cout << int((1 - As.nz / double(As.r * As.c)) * 100) << "%, nnz " << As.nz << std::endl;

        // For now fill A with random numbers between 0-1 and have a random b between 0-1 to avoid
        //   numerical error
        for (int i = 0; i < As.nz; i ++) {
            As.Lx[i] = (Scalar) rand() / (Scalar) (RAND_MAX);
        }

        std::string csv_file = outputFile;

        const int max_bCols = *std::max_element(bColsToTest.begin(), bColsToTest.end());
        const int max_numel_C = As.r * max_bCols;

        SpMMTask<Scalar> spmm_task;
        spmm_task.A = &As;
        spmm_task.B = random_matrix<Scalar>(As.c, max_bCols);
        spmm_task.C = new(std::align_val_t(4096)) Scalar[max_numel_C];
        spmm_task.correct_C = new(std::align_val_t(4096)) Scalar[max_numel_C];
        spmm_task.bRows = As.c;
        spmm_task.filepath = matrixPath;

        auto pattern = spmm_task.A->sparsity_pattern();

        for (const int bCols: bColsToTest) {
            spmm_task.bCols = bCols;

            for (const int nThreads : nThreadsToTest) {
                spmm_task.nThreads = nThreads;

#ifdef MKL
                mkl_set_num_threads(nThreads);
                mkl_set_num_threads_local(nThreads);
                mkl_set_dynamic(0);

                if (nThreads != mkl_get_max_threads()) {
                    std::cerr << "Max threads does not match" << std::endl;
                    exit(-1);
                }
#endif

                std::cout << "Begin Testing, nThreads: " << nThreads << " BCols: " << bCols << std::endl;

                { // compute correct C
                    if (get_method_id_mapping<Scalar>().count(REFERENCE_METHOD) == 0) {
                        std::cerr << "Reference method_id " << REFERENCE_METHOD << " not registered" << std::endl;
                        exit(-1);
                    }
                    c4::yml::ConstNodeRef no_options;
                    additional_options_t no_additional_options;

                    auto baseline_factory = get_method_id_mapping<Scalar>()[REFERENCE_METHOD](no_options);
                    auto verification_baseline = baseline_factory(no_additional_options, spmm_task);
                    (*verification_baseline)();
                    std::memcpy(spmm_task.correct_C, spmm_task.C, spmm_task.cNumel() * sizeof(Scalar));
                }

                std::vector<std::string> method_uids;
                std::map<std::string, csv_row_t> csv_rows;
                std::map<std::string, SpMMFunctor<Scalar>*> executors;
                std::map<std::string, bool> should_tune;
                std::map<std::string, std::string> tuning_grid;
                std::map<std::string, std::string> names;

                // Allow for parallel construction
                omp_set_num_threads(omp_get_num_procs());

                auto construct_method = [&, this](const struct Method& method) {
                    auto name = method.name;

                    auto executor = method.methodFactory(additional_options, spmm_task);
                    ERROR_AND_EXIT_IF(!executor, "Failed to create executor: " << name);

                    std::stringstream ss; ss << (uintptr_t)executor;
                    std::string method_uid = ss.str();

                    #pragma omp critical
                    {
                        if (!method.rowReordering.empty()) {
                            ERROR_AND_EXIT("Row reordering support depricated");
                        }

                        if (!method.tuningParameterGrid.empty()) {
                            should_tune[method_uid] = true;
                            tuning_grid[method_uid] = method.tuningParameterGrid;
                        } else {
                            should_tune[method_uid] = false;
                        }
                    }

                    if (method.config) {
                      executor->set_config(method.config.value());
                    }

                    csv_row_t* csv_row_ptr = nullptr;
                    #pragma omp critical
                    csv_row_ptr = &csv_rows[method_uid];
                    auto& csv_row = *csv_row_ptr;

                    csv_row_insert(csv_row, "name", name);
                    csv_row_insert(csv_row, "matrixPath", matrixPath);
                    csv_row_insert(csv_row, "numThreads", nThreads);

                    // Push into CSV just to make sure it makes it into the header
                    for (const auto &parameter: expand_config_parameters) csv_row_insert(csv_row, parameter, "");

                    // Matrix details
                    csv_row_insert(csv_row, "m", spmm_task.m());
                    csv_row_insert(csv_row, "k", spmm_task.k());
                    csv_row_insert(csv_row, "n", spmm_task.n());
                    csv_row_insert(csv_row, "nnz", spmm_task.A->nz);


                    //  Register Benchmark
                    #pragma omp critical
                    {
                        executors[method_uid] = executor;
                        names[method_uid] = name;
                        method_uids.push_back(method_uid);

                        benchmark::RegisterBenchmark(method_uid.c_str(), [executor](benchmark::State& st) {
                            for (auto _: st) { (*executor)(); }
                        })->Unit(benchmark::kMicrosecond);
                    }

                    if (save_tuning_results) {
                        ERROR_AND_EXIT("Saving tuning results support is deprecated");
                    }
                };


                // Setup the executors
                #pragma omp parallel for
                for (int i = 0; i < methods.size(); i++) {
                    const auto& method = methods[i];
                    if (method.method_id != "mkl") {
                        construct_method(method);
                    }
                }

                omp_set_num_threads(nThreads);
#ifdef MKL
                mkl_set_num_threads(nThreads);
                mkl_set_num_threads_local(nThreads);
                mkl_set_dynamic(0);

                if (nThreads != mkl_get_max_threads()) {
                    std::cerr << "Max threads does not match" << std::endl;
                    exit(-1);
                }
#endif

                // Cant setup inside OMP environment
                for (int i = 0; i < methods.size(); i++) {
                    const auto& method = methods[i];
                    if (method.method_id == "mkl") {
                        construct_method(method);
                    }
                }

                for (auto& [method_uid, executor]: executors) {
                    if (should_tune[method_uid]) {
                        std::string grid_name = tuning_grid[method_uid];

                        if (tuningParameterGrids.count(grid_name) == 0) {
                          std::cerr << "No tuning grid " << grid_name << " defined" << std::endl;
                          exit(-1);
                        }

                        executor->tune(tuningParameterGrids[grid_name], save_tuning_results);
//                        std::cout << "Tuned: " << names[method_uid]
//                                  << " config: " << executor->get_config_rep() << std::endl;
                    }
                }

                // Setup the executors after tuning in parallel
                #pragma omp parallel for
                for (int i = 0; i < method_uids.size(); i++) {
                    const auto& method_uid = method_uids[i];
                    auto& executor = executors[method_uid];
                    executor->setup(); // Setup after the config has been set
                }

                // Test correctness
                for (const auto& [method_uid, executor] : executors) {
                    auto &csv_row = csv_rows[method_uid];

                    zero(spmm_task.C, spmm_task.cNumel());
                    (*executor)();

                    executor->copy_output();
                    auto is_correct = verify<Scalar>(spmm_task);
                    auto is_correct_str = is_correct ? "correct" : "incorrect";

                    if (!is_correct) {
                        std::cout << "Incorrect result for " << csv_row["name"]
                                  << " " << executor->get_config_rep() << std::endl;
                        report_mismatches(spmm_task);
                    }

                    csv_row_insert(csv_row, "correct", is_correct_str);
                }

                //
                // Profiling if requested
                //

                if (profile) {
#ifdef PAPI_AVAILABLE
                    for (const auto& [method_uid, executor] : executors) {
                        auto &csv_row = csv_rows[method_uid];

                        for (int i = 0; i < 3; i++) { (*executor)(); } // Warmup
                        auto profiler = Profiler<SpMMFunctor<Scalar>>(executor);
                        profiler.profile();
                        profiler.log_counters(csv_row);
                    }
#else
                    std::cerr << "WARNING: the profile flag was set but the app was not compiled with PAPI support";
                        std::cerr << " skipping profiling" << std::endl;
#endif
                }

                SavedRunsReporter saved_runs_reporter;
                benchmark::RunSpecifiedBenchmarks(&saved_runs_reporter);
                benchmark::ClearRegisteredBenchmarks();

                for (auto& runs : saved_runs_reporter.latest_runs) {
                    for (auto& run : runs) {
                        const auto &method_uid = run.run_name.str();
                        const auto &name = names[method_uid];

                        auto &executor = *executors[method_uid];
                        auto &csv_row = csv_rows[method_uid];

                        csv_row_insert(csv_row, "time " + run.aggregate_name, run.GetAdjustedRealTime());
                        csv_row_insert(csv_row, "time cpu " + run.aggregate_name, run.GetAdjustedCPUTime());
                        csv_row_insert(csv_row, "iterations", run.iterations);
                    }
                }

#if 1
                double dense_time = 0;
                std::vector<std::pair<std::string, double>> times;
                for (const auto& [method_uid, csv_row] : csv_rows) {
                    auto& name = names[method_uid];

                    if (csv_row.find("time mean") == csv_row.end()) continue;
                    if (name == "MKL_Dense") {
                        dense_time = std::stod(csv_row.at("time median"));
                    }
                    times.push_back({method_uid, std::stod(csv_row.at("time median"))});
                }

                std::sort(times.begin(), times.end(), [](auto &left, auto &right) {
                    return left.second < right.second;
                });

                for (int i = 0; i < std::min((size_t) 30, times.size()); i++) {
                    auto& method_uid = times[i].first;
                    auto& name = names[method_uid];
                    auto& executor = executors[method_uid];

                    std::cout << i + 1 << ". " << times[i].second << " "
                              << name << " "
                              << executor->get_config_rep() << " " << std::endl;
                }

                std::cout << "Dense: " << dense_time << std::endl;
#else
                for (const auto& [method_uid, csv_row] : csv_rows) {
                    std::cout << std::setw(20) << csv_row.at("name") << " ";
                    if (csv_row.find("time cpu mean") != csv_row.end()) {
                        std::cout << std::setw(20) << csv_row.at("time cpu mean") << " ";
                    }

                    if (csv_row.find("time cpu median") != csv_row.end()) {
                        std::cout << std::setw(20) << csv_row.at("time cpu median") << " ";
                    }

                    if (csv_row.find("time cpu stddev") != csv_row.end()) {
                        std::cout << std::setw(20) << csv_row.at("time cpu stddev") << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
#endif
                for (const auto& [method_uid, executor] : executors) {
                    auto &csv_row = csv_rows[method_uid];
                    if (executor == nullptr) continue;

                    // Store Config
                    std::string config_string;
                    config_string += executor->get_config_rep();
                    csv_row_insert(csv_row, "config", config_string);

                    // Store extra info
                    executor->log_extra_info(csv_row);

                    // Cleanup
                    delete executor;
                }

                add_missing_columns(csv_rows);
                write_csv_rows(csv_file, csv_rows);
            }
        }

        // TODO: Move to destructors
        ::operator delete[] (spmm_task.B, std::align_val_t(4096));
        ::operator delete[] (spmm_task.C, std::align_val_t(4096));
        ::operator delete[] (spmm_task.correct_C, std::align_val_t(4096));

        for (auto const& dist : rowDistanceMeasures) { delete dist.second; }

        //exit(-1);
        return 0;
    }

public:
    SpMMExperiment(const ryml::NodeRef& config, std::vector<std::string> search_dirs):
        cake_cntx(cake_query_cntx()) {
        std::cout << "Detected cache sizes " << cake_cntx->L2/1024 << "kb ";
        std::cout << cake_cntx->L3/1024 << "kb" << std::endl;

        bool parse_b_cols_from_filelist = false;
        bool filelist_has_additional_options = false;
        bool using_filelist = false;

        if (config.has_child("options")) {
            auto options = config["options"];
#define UNPACK_OPTION(x, out) if (options.has_child(x)) options[x] >> out;
            UNPACK_OPTION("profile", profile);
            UNPACK_OPTION("output_file", outputFile);
            UNPACK_OPTION("expand_config_parameters", expand_config_parameters);
            UNPACK_OPTION("save_tuning_results", save_tuning_results);
            UNPACK_OPTION("filelist_has_additional_options", filelist_has_additional_options);
#undef UNPACK_OPTION

            if (options.has_child("b_cols")) {
                auto b_cols = options["b_cols"];
                if (b_cols.type() & ryml::SEQ) {
                    options["b_cols"] >> globalBColsToTest;
                } else if (b_cols.type() & ryml::VAL) {
                    int tmp; b_cols >> tmp;
                    if (tmp > 0) {
                        globalBColsToTest = {tmp};
                    } else {
                        parse_b_cols_from_filelist = true;
                    }

                } else {
                    std::cerr << "Unsupported b_col structure" << std::endl;
                    exit(-1);
                }
            }

            if (options.has_child("n_threads")) {
                auto n_threads = options["n_threads"];
                if (n_threads.type() & ryml::SEQ) {
                    options["n_threads"] >> nThreadsToTest;
                } else if (n_threads.type() & ryml::VAL) {
                    int tmp; n_threads >> tmp;
                    nThreadsToTest = { tmp };
                } else {
                    std::cerr << "Unsupported n_threads structure" << std::endl;
                    exit(-1);
                }
            }
        }

        if (config.has_child("matrices")) {
            auto matrices = config["matrices"];
            if (parse_b_cols_from_filelist && !matrices.has_child("filelist")) {
                std::cerr << "If b_cols is specified as -1, ";
                std::cerr << "a filelist must be provided" << std::endl;
                exit(-1);
            }

            if (matrices.has_child("filelist")) {
                std::string filelist; matrices["filelist"] >> filelist;
                using_filelist = true;
                parse_filelist(filelist, search_dirs,
                               parse_b_cols_from_filelist,
                               filelist_has_additional_options);
            }

            if (matrices.has_child("paths")) {
                ERROR_AND_EXIT_IF(parse_b_cols_from_filelist ||
                                  filelist_has_additional_options,
                                  "Cannot paths with options "
                                  "parse_b_cols_from_filelist or "
                                  "filelist_has_additional_options");
                parse_matrix_paths(matrices["paths"], search_dirs);
            }
        }

        for(c4::yml::ConstNodeRef n : config["methods"].children()) {
            methods.push_back(parse_method(n));
        }

        if (config.has_child("row_reorderings")) {
            for (c4::yml::ConstNodeRef n: config["row_reorderings"].children()) {
                auto def = parse_row_reordering_definition(n);
                rowReorderingDefinitions[def.name] = def;
            }
        }

        ERROR_AND_EXIT_IF((parse_b_cols_from_filelist ||
                           filelist_has_additional_options) && !using_filelist,
                          "Cannot use options parse_b_cols_from_filelist or "
                          "filelist_has_additional_options without a filelist");

        if (config.has_child("tuning")) {
            auto tuning = config["tuning"];
            if (tuning.has_children() && tuning.has_child("parameter_grids")) {
                parse_parameter_grids(tuning["parameter_grids"]);
            }
        }
    }

    ~SpMMExperiment() {
      delete cake_cntx;
    }

    int operator()() {
        int argc = 5;
        char strs[argc][256];
        std::strcpy(strs[0], "SpMM_Demo");
        std::strcpy(strs[1], "--benchmark_display_aggregates_only=true");
        std::strcpy(strs[2], "--benchmark_report_aggregates_only=true");
        std::strcpy(strs[3], "--benchmark_repetitions=7");
        std::strcpy(strs[4], "--benchmark_min_time=0.015");
        char* argv[] = {strs[0], strs[1], strs[2], strs[3], strs[4]};

        benchmark::Initialize(&argc, argv);
        benchmark::ReportUnrecognizedArguments(argc, argv);

        if (matricesToTest.size() == 0) {
            std::cerr << "No matrices listed, please either supply matrices:{  filelist: path } ";
            std::cerr << "or matrices: paths: [path, path, ...]" << std::endl;
            return -1;
        }

        if (!perMatrixBColsToTest.empty()) {
            if (perMatrixBColsToTest.size() != matricesToTest.size()) {
                std::cerr << "Number of matrices to test must match number of b_cols to test" << std::endl;
                return -1;
            }
        }

        for (int i = 0; i < matricesToTest.size(); i++) {
//            Memory Leak Debugging
//            static size_t max_allocated = 0;
//            auto mal_info = mallinfo();
//            if (mal_info.fordblks > max_allocated)
//                max_allocated = mal_info.fordblks;
//            std::cout << "Allocated " << mal_info.fordblks << " ["  << max_allocated <<  "] bytes" << std::endl;

            auto bColsToTest =  globalBColsToTest;
            if (!perMatrixBColsToTest.empty()) {
                bColsToTest = { perMatrixBColsToTest[i] };
            }

            additional_options_t additional_options;

            if (additionalOptions.size() > i) {
                additional_options = additionalOptions[i];
            }

            int retVal = run_sample(matricesToTest[i], additional_options, bColsToTest);
            if (retVal) return retVal;
        }

        benchmark::Shutdown();
        return 0;
    }
};

/***********************************************************
 *  File Parses
 **********************************************************/

/** load a file from disk and return a newly created CharContainer */
template<class CharContainer>
size_t file_get_contents(const char *filename, CharContainer *v)
{
    ::FILE *fp = ::fopen(filename, "rb");
    C4_CHECK_MSG(fp != nullptr, "could not open file");
    ::fseek(fp, 0, SEEK_END);
    long sz = ::ftell(fp);
    v->resize(static_cast<typename CharContainer::size_type>(sz));
    if(sz)
    {
        ::rewind(fp);
        size_t ret = ::fread(&(*v)[0], 1, v->size(), fp);
        C4_CHECK(ret == (size_t)sz);
    }
    ::fclose(fp);
    return v->size();
}

/** load a file from disk into an existing CharContainer */
template<class CharContainer>
CharContainer file_get_contents(const char *filename)
{
    CharContainer cc;
    file_get_contents(filename, &cc);
    return cc;
}

/***********************************************************
 *  Main
 **********************************************************/

int main(int argc, char *argv[]) {
    auto config = cpp_testbed::parseInput(argc, argv);

    std::string contents = file_get_contents<std::string>(config.experimentPath.c_str());
    ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(contents));
    ryml::NodeRef root = tree.rootref();

    std::string scalar_type = "float";
    if (root.has_child("options") && root["options"].has_child("scalar_type")) {
        root["options"]["scalar_type"] >> scalar_type;
    } else {
        std::cerr << "No scalar_type set, defaulting to float" << std::endl;
    }

    int result = -1;
    std::string experimentDir = std::filesystem::path(config.experimentPath).parent_path().string() + "/";

    if (scalar_type == "float") {
        auto experiment = SpMMExperiment<float>(root, { config.datasetDir, experimentDir });
        result = experiment();
    } else if (scalar_type == "double")  {
        std::cerr << "Double current unsupported by the baselines " << scalar_type << std::endl;
//        auto experiment = SpMMExperiment<double>(root, config.datasetDir);
//        result = experiment();
    } else {
        std::cerr << "Unsupported scalar type " << scalar_type << std::endl;
    }

    return result;
}
