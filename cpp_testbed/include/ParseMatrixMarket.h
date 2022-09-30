//
// Created by cetinicz on 2021-07-07.
//

#ifndef DDT_PARSEMATRIXMARKET_H
#define DDT_PARSEMATRIXMARKET_H


#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Matrix.h"


namespace cpp_testbed {
namespace MatrixMarket {

template<typename Scalar, typename T>
void copySymLibMatrix(Matrix<Scalar> &m, T symLibMat) {
  // Convert matrix back into regular format
  m.nz = symLibMat->nnz;
  m.r = symLibMat->m;
  m.c = symLibMat->n;

  delete m.Lp;
  delete m.Lx;
  delete m.Li;

  m.Lp = new int[m.r + 1]();
  m.Lx = new double[m.nz]();
  m.Li = new int[m.nz]();

  std::copy(symLibMat->p, symLibMat->p + symLibMat->m + 1, m.Lp);
  std::copy(symLibMat->i, symLibMat->i + symLibMat->nnz, m.Li);
  std::copy(symLibMat->x, symLibMat->x + symLibMat->nnz, m.Lx);
}

template<class Matrix>
Matrix readSparseMatrix(const std::string &path) {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) {
    std::cout << "File " << path << " could not be found..." << std::endl;
    exit(1);
  }
  RawMatrix mat;

  int rows = 0, cols = 0, nnz = 0;
  std::string line;
  bool parsed = false;
  bool sym = false;
  if (file.is_open()) {
    std::stringstream ss;
    std::getline(file, line);
    ss << line;
    // Junk
    std::getline(ss, line, ' ');
    // Matrix
    std::getline(ss, line, ' ');
    // Type
    std::getline(ss, line, ' ');
    if (line != "coordinate") {
      std::cout << "Can only process real matrices..." << std::endl;
      exit(1);
    }
    std::getline(ss, line, ' ');

    // Symmetric
    std::getline(ss, line, ' ');
    if (line == "symmetric") {
      sym = true;
    }

    ss.clear();

    while (std::getline(file, line)) {
      if (line[0] == '%') { continue; }
      if (!parsed) {
        ss << line;
        ss >> rows >> cols >> nnz;
        parsed = true;
        mat.reserve(sym ? nnz * 2 - rows : nnz);
        ss.clear();
        break;
      }
    }
    for (int i = 0; i < nnz; i++) {
      std::getline(file, line);
      std::tuple<int, int, double> t;
      ss << line;
      int row, col;
      double value;
      ss >> row >> col >> value;
      mat.emplace_back(std::make_tuple(row - 1, col - 1, value));
      if (sym && col != row) {
        mat.emplace_back(std::make_tuple(col - 1, row - 1, value));
      }
      ss.clear();
    }
  }
  file.close();

#ifdef METISA
  auto ccc = CSC( rows, cols, sym ? nnz*2-rows : nnz, mat);
  if (std::is_same_v<type, CSR>) {
      return reorderSparseMatrix<CSR>(ccc);
  } else if (std::is_same_v<type, CSC>) {
      return reorderSparseMatrix<CSC>(ccc);
  } else {
      throw std::runtime_error("Error: Matrix storage format not supported");
  }
#endif

  if constexpr(std::is_same_v<Matrix, CSR<typename Matrix::Scalar>>) {
    return CSR<typename Matrix::Scalar>(rows, cols, sym ? nnz * 2 - rows : nnz, mat);
  } else if constexpr(std::is_same_v<Matrix, CSC<typename Matrix::Scalar>>) {
    return CSC<typename Matrix::Scalar>(rows, cols, sym ? nnz * 2 - rows : nnz, mat);
  } else {
    throw std::runtime_error("Error: Matrix storage format not supported");
  }
}

} // namespace MatrixMarket
} // namespace DDT

#endif  //DDT_PARSEMATRIXMARKET_H
