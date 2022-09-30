//
// Created by lwilkinson on 5/26/22.
//

#include <Matrix.h>
#include <boost/dynamic_bitset.hpp>
#include <vector>

template<typename Scalar>
void row_reordering_hamming_distance(CSR<Scalar> &matrix) {
    std::vector<boost::dynamic_bitset<>> row_bitsets(matrix.r);
    std::cout << row_bitsets[0].size() << std::endl;
}