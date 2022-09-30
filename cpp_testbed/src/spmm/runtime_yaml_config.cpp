//
// Created by lwilkinson on 6/5/22.
//
#include "runtime_yaml_config.h"

#include "HammingRowDistance.h"
#include "SpMMELL.h"
#include "baselines/SpMMGeneric.h"
#include "SpMM_ASpT.h"
#include "SpMM_DCSB.h"
#include "SpMM_GECSB.h"
#include "SpMM_Dense.h"
#include "SpMM_MKL.h"
#include "SpMM_TACO.h"
#include "SpMM_SOP.h"
#include "ryml.hpp"
#include "ryml_std.hpp"
#include "spmm.h"

#include "mapping_to_executor.h"

using namespace cpp_testbed;

std::map<std::string, distance_factory_t> distance_mapping = {
        {"hamming", [](SparsityPattern& pattern) { return new HammingRowDistance(pattern); }}
};

std::map<std::string, row_reordering_algo_t*> algo_mapping = {
        {"greedy", greedy_row_reordering}
};


// Util, todo cleanup
using CSRTypes = sop::CSRStorageTypes<int, int>;
using NO_PACKING = sop::PackingDesc<sop::NO_PACKING, sop::NO_PACKING>;
using C_PARTIALY_PACKED = sop::PackingDesc<sop::PARTIAL_PACKING, sop::NO_PACKING>;

template<typename S>
method_mapping_t<S>& get_method_id_mapping() {
    static std::map <std::string, method_factory_factory_t<S>> method_id_mapping = {

    }
};


#if 0

template<typename S>
method_mapping_t<S>& get_method_id_mapping() {
    // NOLINTBEGIN
    #define SPMM_GENERIC_WITH_VEC_WIDTH(id, func, Config)                                                          \
        {id, [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {                                            \
            int i; options["vec_width"] >> i;                                                                      \
            std::string storage_config = "int_int";                                                                \
            if (options.has_child("storage_config")) { options["storage_config"] >>  storage_config; }             \
            return [i, storage_config](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {       \
                if (storage_config == "int_int") {                                                                 \
                    if (i == 512) return new SpMMGeneric<StorageTypes<S, int, int>, Config>                        \
                        (task, &func<StorageTypes<S, int, int>, 512>);                                             \
                    if (i == 256) return new SpMMGeneric<StorageTypes<S, int, int>, Config>                        \
                        (task, &func<StorageTypes<S, int, int>,  256>);                                            \
                } else if (storage_config == "uint8_t_uint16_t") {                                                 \
                    if (i == 512) return new SpMMGeneric<StorageTypes<S, uint16_t, uint8_t>, Config>               \
                        (task, &func<StorageTypes<S, uint16_t, uint8_t>, 512>);                                    \
                    if (i == 256) return new SpMMGeneric<StorageTypes<S, uint16_t, uint8_t>, Config>               \
                        (task, &func<StorageTypes<S, uint16_t, uint8_t>, 256>);                                    \
                } else if (storage_config == "uint8_t_int") {                                                      \
                    if (i == 512) return new SpMMGeneric<StorageTypes<S, int, uint8_t>, Config>                    \
                        (task, &func<StorageTypes<S, int, uint8_t>, 512>);                                         \
                    if (i == 256) return new SpMMGeneric<StorageTypes<S, int, uint8_t>, Config>                    \
                        (task, &func<StorageTypes<S, int, uint8_t>, 256>);                                         \
                }                                                                                                  \
                return nullptr;                                                                                    \
            };                                                                                                     \
        }}
    // NOLINTEND


    static std::map<std::string, method_factory_factory_t<S>> method_id_mapping = {
        SPMM_GENERIC_WITH_VEC_WIDTH("csr_c_1d",   spmm_csr_c,       CSR_C_Config),
        SPMM_GENERIC_WITH_VEC_WIDTH("csr_c_2d",   spmm_csr_c_2d,    CSR_C_2D_Config),
        SPMM_GENERIC_WITH_VEC_WIDTH("csr_c_2d_b", spmm_csr_c_2d_b,  CSR_C_2D_B_Config),
        SPMM_GENERIC_WITH_VEC_WIDTH("csr_a_2d",   spmm_csr_a_2d,    CSR_A_2D_Config),
        {"mkl", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            bool inspector; options["inspector"] >> inspector;
            return [inspector](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
                if (inspector) return new SpMMMKL<S, true >(task);
                else           return new SpMMMKL<S, false>(task);
            };
        }},
        {"mkl_dense", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            return [](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
                return new SpMMMKLDense<S>(task);
            };
        }},
        {"ell", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            int i; options["vec_width"] >> i;
            return [i](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
                if (i == 512) return new SpMMELL<S, 512>(task);
                if (i == 256) return new SpMMELL<S, 256>(task);
                return nullptr;
            };
        }},
        {"gecsb", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
           int i; options["vec_width"] >> i;
           std::string storage; options["storage"] >> storage;
           return [storage, i](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
             if (storage == "CSR") {
               if (i == 512) return new SpMM_GECSB<512, S, int, GECSB_CSR>(task);
               if (i == 256) return new SpMM_GECSB<256, S, int, GECSB_CSR>(task);
               return nullptr;
             } else if (storage == "CSC") {
               if (i == 512) return new SpMM_GECSB<512, S, int, GECSB_CSC>(task);
               if (i == 256) return new SpMM_GECSB<256, S, int, GECSB_CSC>(task);
             }
             return nullptr;
           };
        }},

        {"taco", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            return [](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
                return new SpMMTACO<S>(task);
            };
        }},

        {"dcsb", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            int i; options["vec_width"] >> i;
            bool uint8_row_ptrs; options["8bit_row_ptrs"] >> uint8_row_ptrs;
            std::string storage; options["storage"] >> storage;
            return [storage, i, uint8_row_ptrs](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
                if (storage == "CSR") {
                    if (i == 512 && uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint8_t, DCSB_CSR>(task);
                    if (i == 256 && uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint8_t, DCSB_CSR>(task);
                    if (i == 512 && !uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint16_t, DCSB_CSR>(task);
                    if (i == 256 && !uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint16_t, DCSB_CSR>(task);
                    return nullptr;
                } else if (storage == "CSC") {
                    if (i == 512 && uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint8_t, DCSB_CSC>(task);
                    if (i == 256 && uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint8_t, DCSB_CSC>(task);
                    if (i == 512 && !uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint16_t, DCSB_CSC>(task);
                    if (i == 256 && !uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint16_t, DCSB_CSC>(task);
                } else if (storage == "HYBCSR") {
                    if (i == 512 && uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint8_t, DCSB_HYBCSR>(task);
                    if (i == 256 && uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint8_t, DCSB_HYBCSR>(task);
                    if (i == 512 && !uint8_row_ptrs) return new SpMM_DCSB<S, 512, uint16_t, DCSB_HYBCSR>(task);
                    if (i == 256 && !uint8_row_ptrs) return new SpMM_DCSB<S, 256, uint16_t, DCSB_HYBCSR>(task);
                }
                return nullptr;
            };
        }},
        {"nano", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
            std::string arch; options["arch"] >> arch;
            std::string mapping_id; options["mapping_id"] >> mapping_id;

            int nr = -1;
            if (options.has_child("nr"))
              options["nr"] >> nr;

            bool packed = false;
            if (options.has_child("packed"))
              options["packed"] >> packed;

            bool load_balance = false;
            if (options.has_child("load_balance"))
              options["load_balance"] >> load_balance;

            return [arch, nr, packed, load_balance, mapping_id](
              additional_options_t options, SpMMTask<S>& task)
                -> SpMMFunctor<S>* {

              std::string mapping_id_mut = mapping_id;

              if (mapping_id == "filelist") {
                mapping_id_mut = options["mapping_id"];
              }

              ERROR_AND_EXIT_IF(packed && load_balance,
                "Packing and load-balance are not currently "
                "supported together");

              if (load_balance) {
                return new SpMM_SOP<sop::KDFloatNoPackingLoadBalanced>
                  (get_executor_id(mapping_id_mut, arch, nr), mapping_id_mut, task);
              } else if (packed) {
                return new SpMM_SOP<sop::KDFloatCPartialPacking>
                  (get_executor_id(mapping_id_mut, arch, nr), mapping_id_mut, task);
              } else {
                return new SpMM_SOP<sop::KDFloatNoPacking>
                  (get_executor_id(mapping_id_mut, arch, nr), mapping_id_mut, task);
              }

              return nullptr;
            };
        }},
    };

    return method_id_mapping;
}

#endif

template method_mapping_t<float>& get_method_id_mapping<float>();
// Not yet supported
//template method_mapping_t<double>& get_method_id_mapping<double>();