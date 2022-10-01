#pragma once

#include "spmm/spmm_config.h"
#include <optional>

struct NullConfig: ConfigBase {};

template<typename _Scalar, typename _IndexPtr, typename _Index>
struct StorageTypes {
    using Scalar = _Scalar;
    using IndexPtr = _IndexPtr;
    typedef _Index Index;
};

template<typename StorageTypes, typename Config = NullConfig>
using bspmm_t = void(
        int, int, int,   // C (m x n) = A (m x k) * B (k x n)
        int,           // nnz
        const int *,   // row_indices
        const typename StorageTypes::Scalar *, // values
        const typename StorageTypes::IndexPtr *,  // row_offsets
        const typename StorageTypes::Index *,  // column_indices
        const typename StorageTypes::Scalar *, // dense_matrix
        typename StorageTypes::Scalar *,       // out
        int batch_size,
        const Config &config
);

/********************************************
 *    Implementation Configs
 ********************************************/

struct TiledConfig: ConfigBase {
    int mTile = 16;
    int nTile = 128;

    REGISTER_PARAMS_BEGIN(TiledConfig)
    REGISTER_PARAM("m_tile", mTile),
    REGISTER_PARAM("n_tile", nTile),
    REGISTER_PARAMS_END()
};

struct CSR_A_2D_Config: ConfigBase {
    int mTile = 16;
    int kTile = 16;
    int nTile = 128;

    REGISTER_PARAMS_BEGIN(CSR_A_2D_Config)
    REGISTER_PARAM("m_tile", mTile),
    REGISTER_PARAM("k_tile", kTile),
    REGISTER_PARAM("n_tile", nTile),
    REGISTER_PARAMS_END()
};

struct CSR_C_Config: ConfigBase {
    int nrTile = 32;

    REGISTER_PARAMS_BEGIN(CSR_C_Config)
    REGISTER_PARAM("nr_tile", nrTile),
    REGISTER_PARAMS_END()
};


struct CSR_C_2D_Config: ConfigBase {
    int mTile = 16;
    int nTile = 128;

    REGISTER_PARAMS_BEGIN(CSR_C_2D_Config)
    REGISTER_PARAM("m_tile", mTile),
    REGISTER_PARAM("n_tile", nTile),
    REGISTER_PARAMS_END()
};

struct CSR_C_2D_B_Config: ConfigBase {
    int mTile = 64;
    int kTile = 64;
    int nTile = 64;

    REGISTER_PARAMS_BEGIN(CSR_C_2D_B_Config)
    REGISTER_PARAM("m_tile", mTile),
    REGISTER_PARAM("k_tile", kTile),
    REGISTER_PARAM("n_tile", nTile),
    REGISTER_PARAMS_END()
};

/********************************************
 *    Implementation Signatures
 ********************************************/

typedef struct StorageTypes<float, int, int>            storage_config_1;
typedef struct StorageTypes<float, uint16_t, uint8_t>   storage_config_2;
typedef struct StorageTypes<float, int, uint8_t>        storage_config_3;

#define INSTANTIATE_FOR_ALL_STORAGE_CONFIGS(name, Config) \
    template bspmm_t<storage_config_1, Config> name<storage_config_1, 512>; \
    template bspmm_t<storage_config_1, Config> name<storage_config_1, 256>; \
    template bspmm_t<storage_config_2, Config> name<storage_config_2, 512>; \
    template bspmm_t<storage_config_2, Config> name<storage_config_2, 256>; \
    template bspmm_t<storage_config_3, Config> name<storage_config_3, 512>; \
    template bspmm_t<storage_config_3, Config> name<storage_config_3, 256>

template<typename StorageTypes, int vector_width>
bspmm_t<StorageTypes> spmm_csr_a;
template<typename StorageTypes, int vector_width>
bspmm_t<StorageTypes, CSR_A_2D_Config> spmm_csr_a_2d;
template<typename StorageTypes, int vector_width>
bspmm_t<StorageTypes, CSR_C_Config> spmm_csr_c;
template<typename StorageTypes, int vector_width>
bspmm_t<StorageTypes, CSR_C_2D_Config> spmm_csr_c_2d;
template<typename StorageTypes, int vector_width>
bspmm_t<StorageTypes ,CSR_C_2D_B_Config> spmm_csr_c_2d_b;