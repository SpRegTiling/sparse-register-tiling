//
// Created by lwilkinson on 5/27/22.
//

#include <iostream>
#include <filesystem>
#include <fstream>

#include "cxxopts.hpp"
#include "HammingRowDistance.h"
#include "OverlapPctRowDistance.h"
#include "SparseMatrixIO.h"
#include "csv_log_io.h"

using namespace cpp_testbed;

void analyze(std::string outputFile, std::string matrixPath) {
    typedef cpp_testbed::CSR<float> CSR;
    auto A = cpp_testbed::readSparseMatrix<CSR>(matrixPath);
    auto sparsity_pattern = A.sparsity_pattern();

    std::vector<RowDistance*> distance_measures = {
        static_cast<RowDistance*>(new HammingRowDistance(sparsity_pattern)),
        static_cast<RowDistance*>(new OverlapPctRowDistance(sparsity_pattern))
    };

    int PANEL_SIZE = 16;

    for (auto const& distance_measure : distance_measures) {
        double total_panel_distance = 0;
        int num_panels = 0;

        for (int i = 0; i < A.r; i += PANEL_SIZE) {
            num_panels += 1;
            auto dist = distance_measure->panel_dist(i, std::min(A.r, i + PANEL_SIZE));
            total_panel_distance += dist;
        }

        csv_row_t csv_row;

        csv_row_insert(csv_row, "matrixPath", matrixPath);
        csv_row_insert(csv_row, "distance measure", distance_measure->name());
        csv_row_insert(csv_row, "average panel distance", total_panel_distance / num_panels);
        csv_row_insert(csv_row, "total panel distance", total_panel_distance);
        csv_row_insert(csv_row, "num panels", num_panels);
        csv_row_insert(csv_row, "panel height", PANEL_SIZE);

        // Matrix Details
        csv_row_insert(csv_row, "rows", A.r);
        csv_row_insert(csv_row, "cols", A.c);
        csv_row_insert(csv_row, "nnz", A.nz);

        write_csv_row(outputFile, csv_row);
    }
    // TODO: Move to destructors
    for (auto &distance_measure: distance_measures) { delete distance_measure; }
}

int main(int argc, char *argv[]) {

    cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

    options.add_options()
            ("h,help", "Prints help text")
            ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>()->default_value(""))
            ("f,file_list", "Path a file containing a list of matrices to process.",
             cxxopts::value<std::string>()->default_value(""))
            ("d,dataset_dir", "Path a folder containing the dataset(s).",
             cxxopts::value<std::string>()->default_value(""))
            ("o,output", "Output file", cxxopts::value<std::string>()->default_value("distances.csv"));

    auto result = options.parse(argc, argv);

    auto filelistPath = result["file_list"].as<std::string>();
    auto matrixPath = result["matrix"].as<std::string>();
    auto datasetDir = result["dataset_dir"].as<std::string>();
    auto outputFile = result["output"].as<std::string>();

    if (!filelistPath.empty()) {
        std::ifstream file(filelistPath);
        std::string matrixPath;
        std::string filelistDir = std::filesystem::path(filelistPath).parent_path().string() + "/";

        if (!file.is_open()) {
            std::cout << "Failed to open " << filelistPath << std::endl;
            exit(-1);
        }

        while (std::getline(file, matrixPath)) {
            // Filepath construction priority
            //  1. if path is an absolute path use as is
            //  2. if datasetDir is set use datasetDir + path
            //  3. if datasetDir is not set construct path relative to the filelist file
            if (matrixPath[0] != '/') {
                if (!datasetDir.empty())
                    matrixPath = datasetDir + "/" + matrixPath;
                else
                    matrixPath = filelistDir + matrixPath;
            }

            // Clean path string
            matrixPath = std::filesystem::path(matrixPath).lexically_normal();

            matrixPath = matrixPath;
            std::cout << matrixPath << std::endl;
            analyze(outputFile, matrixPath);
        }

        file.close();
    } else {
        analyze(outputFile, matrixPath);
    }
}