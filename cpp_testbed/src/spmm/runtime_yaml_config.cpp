//
// Created by lwilkinson on 6/5/22.
//
#include "runtime_yaml_config.h"

using namespace cpp_testbed;

#if defined(SPMM_ASpT_ENABLED) && SPMM_ASpT_ENABLED
#include "ASpT/runtime_yaml_config.h"
#endif

#if defined(SPMM_MKL_ENABLED) && SPMM_MKL_ENABLED
#include "MKL/runtime_yaml_config.h"
#endif

#if defined(SPMM_baselines_ENABLED) && SPMM_baselines_ENABLED
#include "baselines/runtime_yaml_config.h"
#endif

#if defined(SPMM_GECSB_ENABLED) && SPMM_GECSB_ENABLED
#include "GECSB/runtime_yaml_config.h"
#endif

#if defined(SPMM_nano_ENABLED) && SPMM_nano_ENABLED
#include "nano/runtime_yaml_config.h"
#endif

#if defined(SPMM_XNN_ENABLED) && SPMM_XNN_ENABLED
#include "XNN/runtime_yaml_config.h"
#endif

template<typename S>
method_mapping_t<S>& get_method_id_mapping() {
    static std::map <std::string, method_factory_factory_t<S>> method_id_mapping = {
#if defined(SPMM_ASpT_ENABLED) && SPMM_ASpT_ENABLED
#include "ASpT/runtime_yaml_config.register"
,
#endif
#if defined(SPMM_MKL_ENABLED) && SPMM_MKL_ENABLED
#include "MKL/runtime_yaml_config.register"
,
#endif
#if defined(SPMM_baselines_ENABLED) && SPMM_baselines_ENABLED
#include "baselines/runtime_yaml_config.register"
,
#endif
#if defined(SPMM_GECSB_ENABLED) && SPMM_GECSB_ENABLED
#include "GECSB/runtime_yaml_config.register"
,
#endif
#if defined(SPMM_nano_ENABLED) && SPMM_nano_ENABLED
#include "nano/runtime_yaml_config.register"
,
#endif
#if defined(SPMM_XNN_ENABLED) && SPMM_XNN_ENABLED
#include "XNN/runtime_yaml_config.register"
,
#endif
    };

    return method_id_mapping;
};


#if 0

template<typename S>
method_mapping_t<S>& get_method_id_mapping() {
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