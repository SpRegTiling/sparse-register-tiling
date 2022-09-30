//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <iostream>

#include "utils/misc.h"
#include "utils/Vec.h"

#include "spmm.h"

#include <vectorclass.h>


using Config = NullConfig;

template<typename StorageTypes, int vector_width>
void spmm_csr_a(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* row_indices,
    const typename StorageTypes::Scalar* __restrict__ values,
    const typename StorageTypes::IndexPtr* __restrict__ row_offsets,
    const typename StorageTypes::Index* __restrict__ column_indices,
    const typename StorageTypes::Scalar* __restrict__ B,
    typename StorageTypes::Scalar* __restrict__ C,
    int batch_size,
    const Config& config
) {
    assert(batch_size == 1);

    using VecType = typename Vec<float, vector_width>::Type;
    using ScalarType = float;

    zero(C, m*n);

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < m; ++i) {
        #pragma GCC unroll 8
        for (int p = row_offsets[i]; p < row_offsets[i+1]; ++p) {
            auto a_val = values[p];
            int b_row = column_indices[p];
            int j;

            VecType aVec(a_val); // Broadcast

            for (j = 0; j < n - (VecType::size() - 1); j += VecType::size()) {
                VecType bVec, cVec;

                bVec.load(&B[(b_row * n) + j]);
                cVec.load(&C[(i * n) + j]);

                cVec = mul_add(aVec, bVec, cVec);

                cVec.store(&C[(i * n) + j]);
            }

            for (; j < n; j++) {
                C[(i * n) + j] += a_val * B[(b_row * n) + j];
            }
        }
    }
}

INSTANTIATE_FOR_ALL_STORAGE_CONFIGS(spmm_csr_a, Config);
