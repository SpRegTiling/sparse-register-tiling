//
// Created by lwilkinson on 11/2/21.
//

#ifndef DDT_SPARSEMATRIXIO_H
#define DDT_SPARSEMATRIXIO_H

#include "ParseMatrixMarket.h"
#include "ParseSMTX.h"

namespace cpp_testbed {

// TODO: Why return matrix here and not just the type
template<class Matrix>
Matrix readSparseMatrix(const std::string &path) {
  std::string file_ext = path.substr(path.find_last_of(".") + 1);

  if (file_ext == "smtx") {
    return cpp_testbed::SMTX::readSparseMatrix<Matrix>(path);
  } else if (file_ext == "mtx") {
    return cpp_testbed::MatrixMarket::readSparseMatrix<Matrix>(path);
  } else {
    throw std::invalid_argument( "Matrix extension not supported" );
  }
}

} // namespace DDT

#endif //DDT_SPARSEMATRIXIO_H
