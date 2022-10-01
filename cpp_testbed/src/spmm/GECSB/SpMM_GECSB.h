//
// Created by lwilkinson on 6/10/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_GECSB_H
#define DNN_SPMM_BENCH_SPMM_GECSB_H

#include <math.h>
#include <vector>


#include "SpMMFunctor.h"
#include "spmm_config.h"

#include "COO.h"
#include "cake_block_dims.h"


#include "utils/error.h"

enum TilingStrategy: int {
  MANUAL_TILING = 0,
  CAKE_TILING = 1,
  CAKE_TILING_WITH_TLB_COMPENSATION = 2,
};

enum GECSB_BLOCK_STORAGE: int {
  GECSB_CSR,
  GECSB_CSC
};

struct GECSBConfig: ConfigBase {
    int mc_tile = 64;
    int kc_tile = 1024;
    int nc_tile = 64;
    int nr_tile = 16;
    int tiling_strategy = 0;

    int beta_10x = 10;
    int sparse_a = 1;
    int max_tlb_entries = 64;
    int tlb_page_size = 4096;

    REGISTER_PARAMS_BEGIN(GECSBConfig)
    REGISTER_PARAM("mc_tile", mc_tile),
    REGISTER_PARAM("kc_tile", kc_tile),
    REGISTER_PARAM("nc_tile", nc_tile),
    REGISTER_PARAM("nr_tile", nr_tile),
    REGISTER_PARAM("tiling_strategy", tiling_strategy),
    REGISTER_PARAM("beta_10x", beta_10x),
    REGISTER_PARAM("sparse_a", sparse_a),
    REGISTER_PARAM("max_tlb_entries", max_tlb_entries),
    REGISTER_PARAM("tlb_page_size", tlb_page_size),
    REGISTER_PARAMS_END()
};

template<typename PtrsType, typename Scalar, int vector_width, enum GECSB_BLOCK_STORAGE storage>
void spmm_csb(
    int m, int k, int n, // C (m x n) = A (m x k) * B (k x n)
    const int* __restrict__      row_indices,
    const std::vector<std::vector<int>>& block_ptrs,
    const PtrsType* __restrict__ b_ptrs,
    const uint16_t* __restrict__ b_inds,
    const Scalar* __restrict__   b_values,
    const Scalar* __restrict__   B,
    Scalar* __restrict__         C,
    int batch_size,
    const GECSBConfig& config
);


template<int vector_width, typename Scalar, typename PtrsType, enum GECSB_BLOCK_STORAGE storage>
class SpMM_GECSB : public SpMMFunctor<Scalar> {
    using Super = SpMMFunctor<Scalar>;

    GECSBConfig config;
    bool packed = false;

    int num_sparse_blocks = 0;
    int num_dense_blocks = 0;
    int nnz_in_sparse_blocks = 0;
    int nnz_in_dense_blocks = 0;

    std::vector<std::vector<int>> block_ptrs;
    PtrsType* b_ptrs         = nullptr;
    uint16_t* b_inds         = nullptr;
    Scalar*   b_values       = nullptr;

    COO<Scalar> *coo = nullptr;

    int num_m_blks = 0;
    int num_k_blks = 0;
    int num_blocks = 0;

    void delete_storage() {
        delete[] b_ptrs;
        delete[] b_inds;
        delete[] b_values;

        b_ptrs = nullptr;
        b_inds = nullptr;
        b_values = nullptr;
    }

//    std::tuple<bool, int, int> pack_block_csc(
//        int ii, int kk,
//        int current_block_offset,
//        int current_block_ptr_offset
//    ) {
//        typename Super::Task& t = this->task;
//
//        PtrsType* _b_col_ptrs = &b_ptrs[current_block_ptr_offset];
//
//        SubmatrixLoc loc = {ii, ii + config.mc_tile, kk, kk + config.kc_tile};
//        for (auto iter = coo->submatrix_begin(loc); iter != coo->submatrix_end(); ++iter) {
//          auto nnz = *iter;
//
//          b_inds[current_block_offset] = nnz.row;
//          b_values[current_block_offset] = nnz.value;
//          current_block_offset++;
//
//          _b_col_ptrs[nnz.col - kk + 1] = current_block_offset;
//        }
//
//        for (int i = 0; i < config.kc_tile; i++) {
//          if (_b_col_ptrs[i + 1] < _b_col_ptrs[i]) {
//            _b_col_ptrs[i + 1] = _b_col_ptrs[i];
//          }
//        }
//
//        return { true, current_block_offset, current_block_ptr_offset + config.kc_tile };
//    }


    std::tuple<bool, int, int> pack_block_csc(
        int ii, int kk,
        int current_block_offset,
        int current_block_ptr_offset
    ) {
      typename Super::Task& t = this->task;

      int b_nnz_cnt = 0;
      int current_nnz_locs[config.mc_tile];
      std::fill(current_nnz_locs, current_nnz_locs + config.mc_tile, -1);

      PtrsType* _b_col_ptrs = &b_ptrs[current_block_ptr_offset];

      for (int i = ii, i_offset = 0; i < std::min(t.m(), ii + config.mc_tile); i++, i_offset++) {
        // Advance to beginning of k-tile
        int p = t.A->Lp[i];
        for (; p < t.A->Lp[i + 1] && t.A->Li[p] < kk; p++) {}

        current_nnz_locs[i_offset] = p;
      }

      for (int k = kk, k_offset = 0; k < std::min(t.k(), kk + config.kc_tile); k++, k_offset++) {
        for (int i = ii, i_offset = 0; i < std::min(t.m(), ii + config.mc_tile); i++, i_offset++) {
          int col_ind = (current_nnz_locs[i_offset] < t.A->Lp[i + 1]) ? t.A->Li[current_nnz_locs[i_offset]] : -1;

          if (col_ind == k) {
            b_inds[current_block_offset] = i;
            b_values[current_block_offset] = t.A->Lx[current_nnz_locs[i_offset]];

            current_nnz_locs[i_offset]++;
            current_block_offset++;
          }

          _b_col_ptrs[k_offset + 1] = current_block_offset;
        }
      }

      for (int i = 0; i < config.kc_tile; i++) {
        if (_b_col_ptrs[i + 1] < _b_col_ptrs[i]) {
          _b_col_ptrs[i + 1] = _b_col_ptrs[i];
        }
      }

      return { true, current_block_offset, current_block_ptr_offset + config.kc_tile };
    }

    std::tuple<bool, int, int> pack_block_csr(
        int ii, int kk,
        int current_block_offset,
        int current_block_ptr_offset
    ) {
        typename Super::Task& t = this->task;
        int nnz_in_block = 0;

        PtrsType* _b_row_ptrs = &b_ptrs[current_block_ptr_offset];
        _b_row_ptrs[0] = current_block_offset;

        SubmatrixLoc loc = {ii, ii + config.mc_tile, kk, kk + config.kc_tile};
        for (auto iter = coo->submatrix_begin(loc); iter != coo->submatrix_end(); ++iter) {
          auto nnz = *iter;

          b_inds[current_block_offset] = nnz.col;
          b_values[current_block_offset] = nnz.value;
          current_block_offset++;

          ERROR_AND_EXIT_IF(
              nnz.row - ii + 1 > config.mc_tile || nnz.row - ii + 1 < 0,
              "Error packing GECSB");
          _b_row_ptrs[nnz.row - ii + 1] = current_block_offset;

          nnz_in_block++;
        }

        for (int i = 0; i < config.mc_tile; i++) {
          if (_b_row_ptrs[i + 1] < _b_row_ptrs[i]) {
            _b_row_ptrs[i + 1] = _b_row_ptrs[i];
          }
        }

        if (nnz_in_block == 0) {
          std::cout << "Empty block: " << ii << ", " << kk << std::endl;
        }

        return { true, current_block_offset, current_block_ptr_offset + config.mc_tile };
    }


    // Use CSR as inner compression
    bool pack_csb() {
        typename Super::Task& t = this->task;

        delete_storage(); // incase a previous tile size was used

        coo = new COO<Scalar>(*t.A);

        int num_m_blks = std::ceil(t.m() / double(config.mc_tile));
        int num_k_blks = std::ceil(t.k() / double(config.kc_tile));
        int num_blocks = num_m_blks * num_k_blks;

        block_ptrs.clear();
        block_ptrs.resize(num_m_blks, std::vector<int>(num_k_blks, 0));

        int ptr_stride = (storage == GECSB_CSR) ? config.mc_tile : config.kc_tile;
        int num_ptrs = num_blocks * ptr_stride;

        b_ptrs     = new PtrsType[num_ptrs + 1]();
        b_inds     = new uint16_t[t.A->nz]();
        b_values   = new Scalar[t.A->nz]();

        int current_block_offset = 0;
        int current_block_ptr_offset = 0;

        for (int ii = 0, tii = 0; ii < t.m(); ii += config.mc_tile, tii++) {
            for (int kk = 0, tkk = 0; kk < t.k(); kk += config.kc_tile, tkk++) {
                block_ptrs[tii][tkk] = current_block_ptr_offset;

                bool status = false;

                if constexpr(storage == GECSB_CSR) {
                    std::tie(status, current_block_offset, current_block_ptr_offset) = \
                        pack_block_csr(ii, kk, current_block_offset, current_block_ptr_offset);
                } else if constexpr(storage == GECSB_CSC){
                    std::tie(status, current_block_offset, current_block_ptr_offset) = \
                        pack_block_csc(ii, kk, current_block_offset, current_block_ptr_offset);
                }

                if (!status) {
                    delete coo;
                    coo = nullptr;
                    return false;
                }
            }
        }

        delete coo;
        coo = nullptr;

        ERROR_AND_EXIT_IF(current_block_offset != t.A->nz, "Error packing GECSB");
        ERROR_AND_EXIT_IF(current_block_ptr_offset != num_ptrs, "Error packing GECSB");

        for (int i = 0; i < num_ptrs; i ++) {
          if (b_ptrs[i + 1] < b_ptrs[i]) {
            ERROR_AND_EXIT("Error packing GECSB");
          }
        }

        packed = true;
        return true;
    }


public:
    SpMM_GECSB(typename Super::Task &task) : Super(task) {
      pack_csb();
      if (config.nr_tile >= task.n()) config.nr_tile = task.n();
    }

    std::string get_config_rep_impl() override {
      return config.rep();
    }

    bool set_config_impl(const typename Super::Config& new_config) override {
        typename Super::Task& t = this->task;

        for (const auto& [k, v] : new_config) {
            config.setVal(k, v);
        }

        if (config.nr_tile >= t.n()) config.nr_tile = t.n();

        if (   (new_config.find("nr_tile") != new_config.end())
            || (new_config.find("nc_tile") != new_config.end())
            || (new_config.find("kc_tile") != new_config.end())
            || (new_config.find("mc_tile") != new_config.end())
            || (new_config.find("tiling_strategy") != new_config.end())) {
            // Tile size changed repack

          if (config.tiling_strategy == CAKE_TILING ||
              config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
            cake_cntx_t* cake_cntx = cake_query_cntx();

            cake_cntx->nr = config.nr_tile;
            cake_cntx->mr = 4;
            cake_cntx->ncores = t.nThreads;

            cache_dims_t* cache_dims = get_cache_dims_4(
                t.m(), t.n(), t.k(), t.nThreads, cake_cntx, KMN,
                nullptr,
                config.sparse_a,
                double(t.A->nz) / (t.m() * t.k()),
                config.beta_10x / 10.0,
                true, true);

            //      cache_dims_t* cache_dims = get_cache_dims_3(
            //          m, b_col_predict, k, num_threads, cake_cntx, KMN,
            //          nullptr, double(coo->nnz()) / (m * k), true, true);

            ERROR_AND_EXIT_IF(!cache_dims->m_c || !cache_dims->k_c || !cache_dims->n_c,
                              "Invalid cache dimensions");

            config.mc_tile = cache_dims->m_c;
            config.kc_tile = cache_dims->k_c;
            config.nc_tile = cache_dims->n_c;

            if (config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {

              int BC_size_bytes = (t.n() * config.kc_tile) * sizeof(Scalar);
              int tlb_entries_used =  BC_size_bytes / config.tlb_page_size;

              if (tlb_entries_used > config.max_tlb_entries) {
                int target_size_bytes = (config.max_tlb_entries * config.tlb_page_size) ;
                int new_k_tile = target_size_bytes / (t.n() * sizeof(Scalar));
                config.kc_tile = new_k_tile;
              }
            }

            free(cake_cntx);
            free(cache_dims);
          }

          packed = false;
        }

        return true;
    }


    ~SpMM_GECSB() {
        delete_storage();
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        typename Super::Task& t = this->task;
        if (!packed) pack_csb();

        spmm_csb<PtrsType, Scalar, vector_width, storage>(
            t.m(), t.k(), t.n(),
            nullptr,
            block_ptrs,
            b_ptrs, b_inds, b_values,
            t.B, t.C, 1,
            config
        );
    }
};


#endif //DNN_SPMM_BENCH_SPMM_GECSB_H
