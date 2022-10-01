//
// Created by lwilkinson on 9/30/22.
//

#pragma once

#include "SpMMGeneric.h"
#include "spmm.h"

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