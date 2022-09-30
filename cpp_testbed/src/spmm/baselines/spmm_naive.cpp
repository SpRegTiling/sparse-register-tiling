#include "spmm.h"

void spmm_naive(
    int m,
    int k,
    int n,
    int nonzeros,
    const int* row_indices,
    const float* values,
    const int* row_offsets,
    const int* column_indices,
    const float* dense_matrix,
    float* output_matrix,
    int batch_size,
    const NullConfig& config) {
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float accumulator = 0.0f;
        for (int l = row_offsets[i]; l < row_offsets[i + 1]; ++l) {
          int column_index = column_indices[l];
          accumulator += values[b * nonzeros + l] *
              dense_matrix[b * k * n + column_index * n + j];
        }
        output_matrix[b * m * n + i * n + j] = accumulator;
      }
    }
  }
}