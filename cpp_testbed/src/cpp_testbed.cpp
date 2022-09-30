/*
 * =====================================================================================
 *
 *       Filename:  DDT.cpp
 *
 *    Description:  File containing main DDT functionality 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 02:15:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */

#define MAX_THREADS 1028

#include "SparseMatrixIO.h"
#include "cpp_testbed.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cpp_testbed {
  int closest_row(int nnz_num, const int *Ap, int init_row = 0){
    int i = init_row;
    while ( Ap[i] <= nnz_num )
      i++;
    return i-1;
  }

  void printTuple(int* t, std::string&& s) {
    std::cout << s << ": (" << t[0] << "," << t[1] << "," << t[2] << ")" << std::endl;
  }
}
