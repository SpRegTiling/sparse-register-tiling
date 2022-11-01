//
// Created by lwilkinson on 5/23/22.
//
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <torch/types.h>


template<typename T>
std::tuple<at::Tensor, at::Tensor> _gen_spmx_trace_csr(T* row_ptrs, T* col_indices, int64_t rows, int64_t cols) {
  const size_t nnz = row_ptrs[rows];
  auto tuples = at::empty({nnz, 3}, at::kInt);
  auto inner_iteration_ptr = at::empty({rows + 1}, at::kInt);

  // NOTE:
  //  nnz num is the same as operation id for SpMV (so we can use p to index
  //  into the tuples tensor)

  // Since nnz number matches op id we can just copy the row_ptrs array
  for (int i = 0; i < rows + 1; i++) { inner_iteration_ptr[i] = row_ptrs[i]; }

  #pragma omp parallel for schedule(dynamic, 128)
  for (int i = 0; i < rows; i++) {
    for (int p = row_ptrs[i]; p < row_ptrs[i+1]; p++) {
      tuples[p][0] = i;
      tuples[p][1] = p;
      tuples[p][2] = col_indices[p];
    }
  }

  return { tuples, inner_iteration_ptr };
}

std::tuple<at::Tensor, at::Tensor> gen_spmx_trace_csr(const at::sparse_csr::SparseCsrTensor& mat) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
  TORCH_CHECK(mat.dim() == 2);

  auto rows = mat.size(0);
  auto cols = mat.size(1);

  auto col_indices = mat.col_indices();
  auto row_ptrs = mat.crow_indices();

  std::tuple<at::Tensor, at::Tensor> ret;

  AT_DISPATCH_INTEGRAL_TYPES(col_indices.type(), "_gen_spmx_trace_csr", ([&] {
     ret = _gen_spmx_trace_csr(
         row_ptrs.data_ptr<scalar_t>(),
         col_indices.data_ptr<scalar_t>(),
         rows, cols
     );
  }));

  return ret;
}

TORCH_LIBRARY_FRAGMENT(ddt_inspector, m) {
  m.def("ddt_inspector::gen_spmx_trace_csr", TORCH_FN(gen_spmx_trace_csr));
}