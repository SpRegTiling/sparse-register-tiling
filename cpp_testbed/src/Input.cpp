/*
 * =====================================================================================
 *
 *       Filename:  Input.cpp
 *
 *    Description:  Parses input for DDT 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 12:06:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */
#include "Input.h"
#include "cpp_testbed.h"

#include <stdlib.h>
#include <assert.h>

#include <iostream>

#include "cxxopts.hpp"

namespace cpp_testbed {
  const int MAX_LIM = 4;

  /**
   * @brief Parses commandline input for the program
   *
   * @param argc 
   * @param argv
   */
  Config parseInput(int argc, char** argv) {
    cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

    options.add_options()
      ("h,help", "Prints help text")
      ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>()->default_value(""))
      ("e,experiment", "Path the experiment config file.", cxxopts::value<std::string>()->default_value(""))
      ("f,file_list", "Path a file containing a list of matrices to process.", cxxopts::value<std::string>()->default_value(""))
      ("d,dataset_dir", "Path a folder containing the dataset(s).", cxxopts::value<std::string>()->default_value(""))
      ("o,output", "Output file", cxxopts::value<std::string>()->default_value("spmm_demo.csv"))
      ("n,numerical_operation", "Numerical operation being performed on matrix.", cxxopts::value<std::string>())
      ("s,storage_format", "Storage format for matrix", cxxopts::value<std::string>()->default_value("CSR"))
      ("t,threads", "Number of parallel threads", cxxopts::value<int>()->default_value("1"))
      ("u,tuning", "Tuning enabled", cxxopts::value<int>()->default_value("0"))
      ("m_tile_size", "Row tile size for SpMM", cxxopts::value<int>()->default_value("64"))
      ("n_tile_size", "Column tile size for SpMM", cxxopts::value<int>()->default_value("64"))
      ("b_matrix_columns", "Number of columns in dense matrix for SpMM", cxxopts::value<int>()->default_value("256"))
      ("p,profile", "Use PAPI to profile")
      ("r,row_reordering", "Include Row Re-ordering Strategies");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    auto matrixPath = result["matrix"].as<std::string>();
    auto filelistPath = result["file_list"].as<std::string>();
    auto experiment = result["experiment"].as<std::string>();
    auto datasetDir = result["dataset_dir"].as<std::string>();
    auto outputFile = result["output"].as<std::string>();
    auto storageFormat = result["storage_format"].as<std::string>();
    auto nThreads = result["threads"].as<int>();
    auto mTileSize = result["m_tile_size"].as<int>();
    auto nTileSize = result["n_tile_size"].as<int>();
    auto bMatrixCols = result["b_matrix_columns"].as<int>();

//    if (filelistPath.empty() && matrixPath.empty()) {
//        std::cout << "Must specify filelist or matrix path" << std::endl;
//        exit(0);
//    }

    NumericalOperation op = OP_SPMM;

    cpp_testbed::StorageFormat sf;
    if (storageFormat == "CSR") {
      sf = cpp_testbed::CSR_SF;
    } else if (storageFormat == "CSC") {
      sf = cpp_testbed::CSC_SF;
    } else {
      std::cout << "'storage_format' must be passed in as one of: ['CSC', 'CSR']" << std::endl;
      exit(0);
    }

    int header = 1;
    if (result.count("no-header")) {
     header = 0;
    }

    return Config { matrixPath, experiment, filelistPath, outputFile, datasetDir,
                    op, header, nThreads, sf,
                    mTileSize, nTileSize, bMatrixCols,
                    result.count("profile") != 0 };
  }
}
