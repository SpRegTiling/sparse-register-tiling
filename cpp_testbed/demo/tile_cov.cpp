#include "cxxopts.hpp"
#include "SparseMatrixIO.h"

#include "COO.h"
#include "TileLocs.h"

/***********************************************************
 *  Main
 **********************************************************/

int main(int argc, char *argv[]) {
    cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

    options.add_options()
            ("h,help", "Prints help text")
            ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>())
            ("i,ti", "Ti", cxxopts::value<int>())
            ("k,tk", "Tk", cxxopts::value<int>());

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
    int ti = args["ti"].as<int>();
    int tk = args["tk"].as<int>();

    typedef CSR<uint8_t> CSR;
    auto A = cpp_testbed::readSparseMatrix<CSR>(matrixPath);
    COO coo(A);

    auto tile_locs = TileLocs({ti, tk}, {coo.rows(), coo.cols()}, TileLocs::COL_FIRST);
    coo.precompute_row_offsets();
    
    int nnz_total = 0;
    int tiles_total = tile_locs.num_i_tiles() * tile_locs.num_j_tiles();
    double nnz_minus_mean_squared_sum = 0.0;

    std::vector<int> working_set_sizes(tiles_total, 0);
    double working_set_size_sum = 0.0;

    #pragma omp parallel for reduction(+ : working_set_size_sum,nnz_total)
    for (int tti = 0; tti < tile_locs.num_i_tiles(); tti++) {
        double working_set_size_sum_local = 0.0;
        int nnz_total_local = 0;
        int tid = tti * tile_locs.num_j_tiles();
        for (auto const& t_loc : tile_locs.row_panel(tti)) {
            auto [nnz, working_set_size] = coo.submatrix_working_set_size(t_loc.loc, 256);
            nnz_total_local += nnz;
            working_set_sizes[tid] = working_set_size;
            working_set_size_sum_local += working_set_size;
            tid++;
        }

        working_set_size_sum += working_set_size_sum_local;
        nnz_total += nnz_total_local;
    }

    double working_set_size_mean = working_set_size_sum / (double)(tiles_total);
    double working_set_size_sq_var_sum = 0.0;
    for (auto const& ws : working_set_sizes) {
        working_set_size_sq_var_sum += (ws - working_set_size_mean) * (ws - working_set_size_mean);
    }

    double working_set_size_sq_var_mean = working_set_size_sq_var_sum / (double)(tiles_total - 1);
    double stddev = std::sqrt(working_set_size_sq_var_mean);

    std::cout << stddev/working_set_size_mean << "," << nnz_total << "," << coo.nnz() << "," << (double) coo.nnz() / (coo.rows() * coo.cols()) << std::endl;
    return 0;
}
