/*
 * =====================================================================================
 *
 *       Filename:  DDT.h
 *
 *    Description:  Header file for DDT.cpp 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 02:16:50 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */

#ifndef DDT_DDT
#define DDT_DDT

#include "ParseMatrixMarket.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpp_testbed {
  enum NumericalOperation {
    OP_SPMV,
    OP_SPTRS,
    OP_SPMM
  };
 
  enum StorageFormat {
    CSR_SF,
    CSC_SF
  };

  struct MemoryTrace {
    int** ip;
    int ips;
  };

  struct Config {
    std::string matrixPath;
    std::string experimentPath;
    std::string filelistPath;
    std::string outputFile;
    std::string datasetDir;
    NumericalOperation op;
    int header;
    int nThread;
    StorageFormat sf;
    int mTileSize;
    int nTileSize;
    int bMatrixCols;
    bool profile;
  };

  void printTuple(int* t, std::string&& s);
}

#endif
