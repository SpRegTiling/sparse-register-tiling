//
// Created by lwilkinson on 6/10/22.
//

#pragma once

#include "utils/Vec.h"
#include "utils/algorithmic.h"

#include "../SpMMFunctor.h"
#include "../MKL/MKL_utils.h"
#include "COO.h"

#include "sop.h"
#include "cake_block_dims.h"

#include <math.h>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>


//#define PACK_B

using namespace cpp_testbed;
using std::vector;

struct SOPConfig: ConfigBase {
    int m_tile = 16;
    int k_tile = 256;
    int n_tile = 64;

    int tiling_strategy = 0;
    int beta_10x = 10;
    int sparse_a = 1;
    int max_tlb_entries = 64;
    int tlb_page_size = 4096;

    REGISTER_PARAMS_BEGIN(SOPConfig)
    REGISTER_PARAM("m_tile", m_tile),
    REGISTER_PARAM("k_tile", k_tile),
    REGISTER_PARAM("n_tile", n_tile),
    REGISTER_PARAM("tiling_strategy", tiling_strategy),
    REGISTER_PARAM("beta_10x", beta_10x),
    REGISTER_PARAM("sparse_a", sparse_a),
    REGISTER_PARAM("max_tlb_entries", max_tlb_entries),
    REGISTER_PARAM("tlb_page_size", tlb_page_size),
    REGISTER_PARAMS_END()
};


template<typename KernelDesc>
class SpMM_SOP : public SpMMFunctor<typename KernelDesc::Scalar> {
    using Config = SOPConfig;

    using Super  = SpMMFunctor<typename KernelDesc::Scalar>;
    using Scalar = typename KernelDesc::Scalar;

    sop::MatMul<KernelDesc>* sop_matmul = nullptr;
    Config config;

    std::string executor_id;
    std::string mapping_id;


public:
    SpMM_SOP(std::string executor_id,
             std::string mapping_id,
             typename Super::Task &task) :
            Super(task),
            executor_id(executor_id),
            mapping_id(mapping_id) {

        typename Super::Task& t = this->task;

        sop::TileConfig tile_config;
        tile_config.N_c = config.n_tile;
        tile_config.M_c = config.m_tile;
        tile_config.K_c = config.k_tile;
        tile_config.tiling_strategy =
            config.tiling_strategy ? sop::CAKE_TILING : sop::MANUAL_TILING;

        delete sop_matmul;
        sop_matmul = nullptr;
    }

//    void log_extra_info(cpp_testbed::csv_row_t& row) override {
//        csv_row_insert(row, "total_tile_count", stats.total_tile_count);
//
//        csv_row_insert(row, "sop_tiles_count", stats.sop_tiles_count);
//        csv_row_insert(row, "sop_tiles_nnz_count", stats.sop_tiles_nnz_count);
//        csv_row_insert(row, "sop_tiles_padding", stats.sop_tiles_padding);
//
//        csv_row_insert(row, "csr_tiles_count", stats.csr_tiles_count);
//        csv_row_insert(row, "csr_tiles_nnz_count", stats.csr_tiles_nnz_count);
//
//        csv_row_insert(row, "dense_tiles_count", stats.dense_tiles_count);
//        csv_row_insert(row, "dense_tiles_padding", stats.dense_tiles_padding);
//        csv_row_insert(row, "dense_tiles_nnz_count", stats.dense_tiles_nnz_count);
//    }

    std::string get_config_rep_impl() override {
        return config.rep();
    }

    bool set_config_impl(const typename Super::Config& new_config) override {
        typename Super::Task& t = this->task;
        auto old_config = config;

        for (const auto& [k, v] : new_config) { config.setVal(k, v); }

        if (old_config.m_tile != config.m_tile
            || old_config.k_tile != config.k_tile
            || old_config.n_tile != config.n_tile
            || old_config.tiling_strategy != config.tiling_strategy
            || old_config.beta_10x != config.beta_10x
            || old_config.sparse_a != config.sparse_a
            || old_config.max_tlb_entries != config.max_tlb_entries
            || old_config.tlb_page_size != config.tlb_page_size) {

            sop::TileConfig tile_config;
            tile_config.N_c = config.n_tile;
            tile_config.M_c = config.m_tile;
            tile_config.K_c = config.k_tile;
            tile_config.beta = float(config.beta_10x) / 10.f;
            tile_config.sparse_a = config.sparse_a;
            tile_config.tiling_strategy = (sop::TilingStrategy) config.tiling_strategy;
            tile_config.tlb_page_size = config.tlb_page_size;
            tile_config.max_tlb_entries = config.max_tlb_entries;

            delete sop_matmul;
            sop_matmul = new sop::MatMul<KernelDesc>(
                t.m(), t.k(), t.n(),
                t.A->Lx, t.A->Lp, t.A->Li,
                tile_config, t.nThreads,
                executor_id,
                mapping_id
            );

            tile_config = sop_matmul->get_config();

            // Save the file config for reporting
            config.n_tile = tile_config.N_c;
            config.m_tile = tile_config.M_c;
            config.k_tile = tile_config.K_c;
            config.tiling_strategy = tile_config.tiling_strategy;

            delete sop_matmul;
            sop_matmul = nullptr;
        }
        return true;
    }

    ~SpMM_SOP() {
        delete sop_matmul;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() override {
        typename Super::Task& t = this->task;
        if (!sop_matmul) {
            sop::TileConfig tile_config;
            tile_config.N_c = config.n_tile;
            tile_config.M_c = config.m_tile;
            tile_config.K_c = config.k_tile;
            tile_config.beta = float(config.beta_10x) / 10.f;
            tile_config.sparse_a = config.sparse_a;
            tile_config.tiling_strategy = (sop::TilingStrategy) config.tiling_strategy;
            tile_config.tlb_page_size = config.tlb_page_size;
            tile_config.max_tlb_entries = config.max_tlb_entries;

            sop_matmul = new sop::MatMul<KernelDesc>(
              t.m(), t.k(), t.n(),
              t.A->Lx, t.A->Lp, t.A->Li,
              tile_config, t.nThreads,
              executor_id,
              mapping_id
            );
        }

        (*sop_matmul)(t.C, t.B, t.n());
    }
};
