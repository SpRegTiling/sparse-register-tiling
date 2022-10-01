//
// Created by lwilkinson on 6/5/22.
//

#ifndef DNN_SPMM_BENCH_EXPERIMENT_MAPPING_H
#define DNN_SPMM_BENCH_EXPERIMENT_MAPPING_H

#include <functional>
#include <map>

#include "ryml.hpp"
#include "SpMMFunctor.h"
#include "Matrix.h"
#include "row_reordering_algos.h"
#include "RowDistance.h"

using additional_options_t = std::map<std::string, std::string>;

#define BASELINE_METHOD "mkl_dense"

// Method mapping
template<typename S> using method_factory_t = std::function<SpMMFunctor<S>* (additional_options_t options, SpMMTask<S>&)>;
template<typename S> using method_factory_factory_t = std::function<method_factory_t<S>(c4::yml::ConstNodeRef options)>;
template<typename S> using method_mapping_t = std::map<std::string, method_factory_factory_t<S>>;
template<typename S> method_mapping_t<S>& get_method_id_mapping();

#endif //DNN_SPMM_BENCH_EXPERIMENT_MAPPING_H
