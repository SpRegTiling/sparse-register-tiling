//
// Created by cetinicz on 2021-10-30.
//

#include <type_traits>
#include <map>
#include <cxxabi.h>
#include <filesystem>
#include <iostream>
#include <iomanip>      // std::setw
//#include <aligned_new>

#include <omp.h>
#include <ryml_std.hpp>
#include <ryml.hpp>

#include "utils/misc.h"
#include "utils/error.h"

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
 *  Experiment
 **********************************************************/

template<typename Scalar>
class SpMMExperiment {
    struct Method {
        std::string name;
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
        return { name, factory, row_reordering, tuning_grid, config };
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

        constexpr int WARMUP_ITERATIONS = 3;
        constexpr int MEASURED_ITERATIONS = 17; // NOTE: Each measured iteration will consist of multiple runs

        typedef CSR<Scalar> CSR;
        auto As = cpp_testbed::readSparseMatrix<CSR>(matrixPath);

        if (As.r % 32 != 0 || As.c % 32 != 0) return 0;

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

#ifdef RASPBERRY_PI
                int runs_per_iteration = 1;
#else
                int runs_per_iteration = std::max((int) std::ceil(2e8 / (spmm_task.A->nz * spmm_task.bCols)), 1);
                runs_per_iteration *= std::max(1, spmm_task.nThreads);
#endif

                std::cout << "Begin Testing, nThreads: " << nThreads << " BCols: " << bCols;
                std::cout << " Runs per iter: " << runs_per_iteration << " nnz_per_bcol " <<  runs_per_iteration << std::endl;

                { // compute correct C
                    if (get_method_id_mapping<Scalar>().count(BASELINE_METHOD) == 0) {
                        std::cerr << "Baseline method_id " << BASELINE_METHOD << " not registered" << std::endl;
                        exit(-1);
                    }
                    c4::yml::ConstNodeRef no_options;
                    additional_options_t no_additional_options;

                    auto baseline_factory = get_method_id_mapping<Scalar>()[BASELINE_METHOD](no_options);
                    auto verification_baseline = baseline_factory(no_additional_options, spmm_task);
                    (*verification_baseline)();
                    std::memcpy(spmm_task.correct_C, spmm_task.C, spmm_task.cNumel() * sizeof(Scalar));
                }

                std::vector<csv_row_t> csv_rows;
                csv_rows.reserve(methods.size());

                double baseline_time = 0;
                for (const auto &method: methods) {
#ifdef MKL
                    // Configure everytime to just make sure nothing gets messed up
                    mkl_set_num_threads(nThreads);
                    mkl_set_num_threads_local(nThreads);
                    mkl_set_dynamic(1);

                    if (nThreads != mkl_get_max_threads()) {
                        std::cerr << "Max threads does not match" << std::endl;
                        exit(-1);
                    }
#endif
                    omp_set_num_threads(nThreads);


                    csv_row_t csv_row;
                    auto name = method.name;

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

                    std::cout << "Running: " << std::setw(40) << std::left << name;
                    std::vector<double> timings(MEASURED_ITERATIONS);

                    auto executor = method.methodFactory(additional_options, spmm_task);
                    ERROR_AND_EXIT_IF(!executor,
                                      "Failed to create executor: " << name);

                    if (!method.rowReordering.empty()) {
                        if (rowReorderingDefinitions.count(method.rowReordering) == 0) {
                            std::cerr << "Row ordering " << method.rowReordering << " not defined, used by ";
                            std::cerr << method.name << std::endl;
                            exit(-1);
                        }

                        if (rowOrderings.count(method.rowReordering) == 0) {
                            auto def = rowReorderingDefinitions[method.rowReordering];
                            if (rowDistanceMeasures.count(def.distance) == 0)
                                rowDistanceMeasures[def.distance] = construct_distance(def.distance, pattern);

                            rowOrderings[def.name] = compute_ordering(def.algo, 16, rowDistanceMeasures[def.distance]);
                        }

                        executor->set_row_reordering(rowOrderings[method.rowReordering]);
                    }

                    zero(spmm_task.C, spmm_task.cNumel());

                    if (!method.tuningParameterGrid.empty()) {
                        if (tuningParameterGrids.count(method.tuningParameterGrid) == 0) {
                            std::cerr << "No tuning grid " << method.tuningParameterGrid << " defined" << std::endl;
                            exit(-1);
                        }

                        for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) { (*executor)(); }
                        executor->tune(tuningParameterGrids[method.tuningParameterGrid], save_tuning_results);
                    } else if (method.config) {
                        executor->set_config(method.config.value());
                    }

                    for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) { (*executor)(); }

#ifdef VTUNE_AVAILABLE
                    std::string event_name =
                        method.name + " n " + std::to_string(bCols) \
                        + " nt " + std::to_string(spmm_task.nThreads);
                    __itt_event mark_event = __itt_event_create( event_name.c_str(), event_name.size() );
                    __itt_resume();
                    __itt_event_start( mark_event );
#endif
                    report_packing_time = true;
                    for (int iter = 0; iter < MEASURED_ITERATIONS; iter++) {
                        sym_lib::timing_measurement t1;

                        t1.start_timer();
                        for (int run = 0; run < runs_per_iteration; run++) (*executor)();
                        t1.measure_elapsed_time();

                        timings[iter] = t1.elapsed_time / runs_per_iteration;
                    }
                    report_packing_time = false;

#ifdef VTUNE_AVAILABLE
                    __itt_event_end( mark_event );
                    __itt_pause();
#endif

                    auto is_correct = verify<Scalar>(spmm_task);
                    auto is_correct_str = is_correct ? "correct" : "incorrect";

                    std::cout << std::setw(12) << std::left << is_correct_str;
                    std::cout << std::setw(20) << std::to_string(median(timings) * 1e6) + " us";

                    if (baseline_time == 0) {
                        baseline_time = median(timings);
                    }

                    std::cout << std::setw(12) << std::left << std::to_string(baseline_time / median(timings)) + "x";

                    if (profile) {
#ifdef PAPI_AVAILABLE
                        auto profiler = Profiler<SpMMFunctor<Scalar>>(executor);
                        profiler.profile();
                        profiler.log_counters(csv_row);
#else
                        std::cerr << "WARNING: the profile flag was set but the app was not compiled with PAPI support";
                        std::cerr << " skipping profiling" << std::endl;
#endif
                    }

                    std::string config_string;
                    if (!method.tuningParameterGrid.empty() || method.config) {
                        config_string += executor->get_config_rep();
                    }


                    std::cout << config_string;
                    std::cout << std::endl;

                    csv_row_insert(csv_row, "tuning", config_string);
                    csv_row_insert(csv_row, "time mean", mean(timings));
                    csv_row_insert(csv_row, "time median", median(timings));
                    csv_row_insert(csv_row, "correct", is_correct_str);

                    if (!is_correct) {
                        report_mismatches(spmm_task);
                    }

                    executor->log_extra_info(csv_row);

                    if (save_tuning_results) {
                        csv_row_insert(csv_row, "tuning", "saved result");
                        for (const auto &[config, time_median, time_mean]: executor->saved_results) {
                            expand_config(csv_row, config);
                            csv_row_insert(csv_row, "time mean", time_mean);
                            csv_row_insert(csv_row, "time median", time_median);
                            csv_rows.push_back(csv_row);
                        }
                    }

                    csv_rows.push_back(std::move(csv_row));
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
