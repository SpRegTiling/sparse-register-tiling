#include "cxxopts.hpp"
#include "SparseMatrixIO.h"

#include "COO.h"
#include "TileLocs.h"
#include "MicroKernelPackerFactory.h"
#include "mapping_io.h"
#include "mapping_to_executor.h"

/***********************************************************
 *  Main
 **********************************************************/

int main(int argc, char *argv[]) {
    cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

    options.add_options()
            ("h,help", "Prints help text")
            ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>())
            ("i,ti", "Ti", cxxopts::value<int>())
            ("k,tk", "Tk", cxxopts::value<int>())
            ("n,nr", "nr", cxxopts::value<int>())
            ("p,mapping", "mapping", cxxopts::value<std::string>());

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (!args.count("matrix") || !args.count("ti") || !args.count("tk")) {
        std::cerr << "Missing required arguments: matrix, ti, tk" << std::endl;
        exit(-1);
    } 

    std::string matrixPath = args["matrix"].as<std::string>();
    int M_c = args["ti"].as<int>();
    int K_c = args["tk"].as<int>();
    int N_r = args["nr"].as<int>();
    std::string mapping_id = args["mapping"].as<std::string>();

    typedef CSR<float> CSR;
    auto A = cpp_testbed::readSparseMatrix<CSR>(matrixPath);
    COO coo(A);


    std::string filepath(__FILE__);
    auto end_of_path = filepath.find_last_of('/');
    filepath = filepath.substr(0, end_of_path + 1);

    auto executor_id = get_executor_id(mapping_id, "AVX512", 512, N_r);
    auto packer_factory = sop::MicroKernelPackerFactory<float>::get_factory(executor_id);
    auto nanokernel_mapping = sop::read_pattern_mapping(mapping_id,
       {"mappings/", filepath + "../mappings/", filepath + "../../spmm_nano_kernels/mappings/"});
    auto packer = packer_factory->create_specialized_packer(nanokernel_mapping);

    auto M_r = packer->M_r;
    coo.pad_to_multiple_of(M_r);

    auto tile_locs = TileLocs({M_c, K_c}, {coo.rows(), coo.cols()}, TileLocs::COL_FIRST);
    Shape matrix_tiled_shape = {tile_locs.num_i_tiles(), tile_locs.num_j_tiles()};

    coo.precompute_row_offsets();

    std::vector<std::vector<sop::PackedTile<float>>> packed_tiles;
    packed_tiles.resize(matrix_tiled_shape.rows);
    for (auto& panel_packed_tiles : packed_tiles){
      panel_packed_tiles.resize(matrix_tiled_shape.cols);
    }

    int nnz_total = 0;
    int total_nano_kernels = 0;
    int m = coo.rows();
    const int panels_per_tile = M_c / M_r;

    //#pragma omp parallel for num_threads(16) schedule(dynamic)
    for (int ti = 0; ti < matrix_tiled_shape.rows; ti++) {
      const auto panel_tile_locs = tile_locs.row_panel(ti);
      for (int tj = 0; tj < panel_tile_locs.size(); tj++) {
          int M_c_tile = std::min(m - M_c * ti, M_c);
          int panels_in_tile = M_c_tile / M_r;

          if (M_c_tile % M_r) {
              std::cerr << "Bad M_c size, M_c_tile: " << M_c_tile << " M_c: " << M_c << ", M_r: " << M_r << std::endl;
              exit(-1);
          }

          auto t_loc = panel_tile_locs[tj].loc;
          packed_tiles[ti][tj].type = sop::SPARSE_SOP;
          packed_tiles[ti][tj].loc = t_loc;
          packed_tiles[ti][tj].shape = t_loc.shape();
          packed_tiles[ti][tj].load_c = tj != 0;
          packed_tiles[ti][tj].free_on_destruction = true;
          packed_tiles[ti][tj].packed_values = false;
          packed_tiles[ti][tj].sop.num_panels = panels_in_tile;
          packed_tiles[ti][tj].sop.panel_descs =
                  new sop::MicroKernelPackedData<float>[panels_per_tile];

          auto panel_descs = packed_tiles[ti][tj].sop.panel_descs;

          int nnz_count = 0;
          for (int panel_id = 0; panel_id < panels_in_tile; panel_id++) {
            int global_panel_id = ti * panels_per_tile + panel_id;
            SubmatrixLoc panel_loc = t_loc;
            panel_loc.rows.start = global_panel_id * M_r;
            panel_loc.rows.end = (global_panel_id + 1) * M_r;

            nnz_count += coo.submatrix_nnz_count(panel_loc);
          }

          nnz_total += nnz_count;

          if (nnz_count == 0) {
              packed_tiles[ti][tj].type = sop::EMPTY_TILE;
              continue;
          }

          for (int panel_id = 0; panel_id < panels_in_tile; panel_id++) {
              int global_panel_id = ti * panels_per_tile + panel_id;
              SubmatrixLoc panel_loc = t_loc;
              panel_loc.rows.start = global_panel_id * M_r;
              panel_loc.rows.end = (global_panel_id + 1) * M_r;
              packer->pack(panel_descs[panel_id], panel_loc, coo, false);
          }
          total_nano_kernels += packed_tiles[ti][tj].count_nano_kernels();
      }
    }

    std::cout << total_nano_kernels << "," << nnz_total << "," << coo.nnz() << "," << (double) coo.nnz() / (coo.rows() * coo.cols()) << "," << std::endl;
    return 0;
}
