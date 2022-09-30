////
//// Created by lwilkinson on 5/23/22.
////
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <torch/types.h>

template<typename T>
std::tuple<at::Tensor, at::Tensor> _compute_consecutive_fopds(
    torch::PackedTensorAccessor32<T, 2> tuples,
    torch::PackedTensorAccessor32<T, 1> inner_iter_ptrs,
    int64_t dim) {
  int outer_iterations = inner_iter_ptrs.size(0) - 1;

  TORCH_CHECK(tuples.size(1) == 3);

  torch::Tensor fopd_inner_iter_ptrs = torch::zeros({outer_iterations + 1}, at::kInt);
  torch::Tensor fopd_tuples = torch::zeros({(int) tuples.size(0), (int) tuples.size(1)}, at::kInt);

  auto fopd_inner_iter_ptrs_p = fopd_inner_iter_ptrs.packed_accessor32<int, 1>();
  auto fopd_tuples_p = fopd_tuples.packed_accessor32<int, 2>();

  auto fopds = torch::zeros({outer_iterations + 1}, at::kInt);

  if (dim == 0) {
    // Pre-compute number of FOPDs per outer iteration
    fopd_inner_iter_ptrs_p[0] = 0;
    for (int i = 0; i < outer_iterations - 1; i ++) {
      auto row_len =  inner_iter_ptrs[i+1] - inner_iter_ptrs[i];
      auto next_row_len =  inner_iter_ptrs[i+2] - inner_iter_ptrs[i+1];
      auto num_fopd = std::min(row_len, next_row_len);
      fopd_inner_iter_ptrs_p[i+1] = num_fopd + fopd_inner_iter_ptrs_p[i];
    }

    // Last row has 0 fopds
    fopd_inner_iter_ptrs_p[outer_iterations] = fopd_inner_iter_ptrs_p[outer_iterations-1];

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < outer_iterations; i ++) {
      auto row_start = inner_iter_ptrs[i];
      auto next_row_start = inner_iter_ptrs[i+1];
      auto num_fopd = fopd_inner_iter_ptrs_p[i+1] - fopd_inner_iter_ptrs_p[i];
      auto fopd_start_offset = fopd_inner_iter_ptrs_p[i];

      T (*tuples_base)[3] = (T (*)[3]) tuples[row_start].data();
      T (*tuples_next)[3] = (T (*)[3]) tuples[next_row_start].data();
      int (*fopd_tuples_base)[3] = (int (*)[3]) fopd_tuples_p[fopd_start_offset].data();

      for (int k = 0; k < num_fopd; k++) {
        fopd_tuples_base[k][0] = tuples_next[k][0] - tuples_base[k][0];
        fopd_tuples_base[k][1] = tuples_next[k][1] - tuples_base[k][1];
        fopd_tuples_base[k][2] = tuples_next[k][2] - tuples_base[k][2];
      }
    }
  } else {
    // Pre-compute number of FOPDs per outer iteration
    fopd_inner_iter_ptrs_p[0] = 0;
    for (int i = 0; i < outer_iterations; i ++) {
      auto num_fopd = inner_iter_ptrs[i + 1] - inner_iter_ptrs[i] - 1;
      if (num_fopd < 0) num_fopd = 0;
      fopd_inner_iter_ptrs_p[i+1] = fopd_inner_iter_ptrs_p[i] + num_fopd;
    }

    /* dim = 1 */
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < outer_iterations; i ++) {
      auto row_start = inner_iter_ptrs[i];
      auto num_fopd = fopd_inner_iter_ptrs_p[i+1] - fopd_inner_iter_ptrs_p[i];
      auto fopd_start_offset = fopd_inner_iter_ptrs_p[i];

      T (*tuples_base)[3] = (T (*)[3]) tuples[row_start].data();
      T (*tuples_next)[3] = (T (*)[3]) tuples[row_start + 1].data();
      int (*fopd_tuples_base)[3] = (int (*)[3]) fopd_tuples_p[fopd_start_offset].data();

      for (int k = 0; k < num_fopd; k++) {
        fopd_tuples_base[k][0] = tuples_next[k][0] - tuples_base[k][0];
        fopd_tuples_base[k][1] = tuples_next[k][1] - tuples_base[k][1];
        fopd_tuples_base[k][2] = tuples_next[k][2] - tuples_base[k][2];
      }
    }
  }

  auto total_fopds = fopd_inner_iter_ptrs_p[outer_iterations];
  fopd_tuples.resize_({total_fopds, tuples.size(1)});
  return { fopd_tuples, fopd_inner_iter_ptrs };
}

std::tuple<at::Tensor, at::Tensor> compute_consecutive_fopds(
    const torch::Tensor& tuples,
    const torch::Tensor& inner_iter_ptrs,
    int64_t dim) {

  TORCH_CHECK(tuples.dim() == 2);
  TORCH_CHECK(inner_iter_ptrs.dim() == 1);

  TORCH_CHECK(dim < 2);
  std::tuple<at::Tensor, at::Tensor> ret;

  AT_DISPATCH_INTEGRAL_TYPES(tuples.type(), "_compute_consecutive_fopds", ([&] {
     ret = _compute_consecutive_fopds(
         tuples.packed_accessor32<scalar_t, 2>(),
         inner_iter_ptrs.packed_accessor32<scalar_t, 1>(),
          dim
     );
  }));

  return ret;
}

template<typename T>
at::Tensor _hist_mine_consecutive_fopds(
    torch::PackedTensorAccessor32<T, 2> fopd_tuples,
    torch::PackedTensorAccessor32<T, 1> fopd_inner_iter_ptrs,
    int64_t access_function,
    int64_t max_consecutive) {
  int outer_iterations = fopd_inner_iter_ptrs.size(0) - 1;
  auto hist = at::zeros({max_consecutive + 1}, at::kInt);

  #pragma omp parallel for schedule(dynamic, 32)
  for (int i = 0; i < outer_iterations; i ++) {
      auto local_hist = at::zeros({max_consecutive + 1}, at::kInt);
      int inner_iteration_end = fopd_inner_iter_ptrs[i+1];
      for (int p = fopd_inner_iter_ptrs[i]; p < inner_iteration_end; p++) {
          int k = 0;
          while (k < max_consecutive
                 && p + k + 1 < inner_iteration_end
                 && fopd_tuples[p+k+1][access_function] == fopd_tuples[p+k][access_function]) {
            k++;
          }

          p += k; // Don't double count
          local_hist[k] += 1;
      }
      #pragma omp critical
      {
          hist += local_hist;
      };
  }


  return hist;
}


at::Tensor hist_mine_consecutive_fopds(
    const torch::Tensor& fopd_tuples,
    const torch::Tensor& fopd_inner_iter_ptrs,
    int64_t access_function,
    int64_t max_consecutive) {

  TORCH_CHECK(fopd_tuples.dim() == 2);
  TORCH_CHECK(fopd_inner_iter_ptrs.dim() == 1);

  TORCH_CHECK(access_function < fopd_tuples.size(1));
  at::Tensor ret;

  AT_DISPATCH_INTEGRAL_TYPES(fopd_tuples.type(), "_hist_mine_consecutive_fopds", ([&] {
     ret = _hist_mine_consecutive_fopds(
       fopd_tuples.packed_accessor32<scalar_t, 2>(),
       fopd_inner_iter_ptrs.packed_accessor32<scalar_t, 1>(),
       access_function,
       max_consecutive
     );
  }));

  return ret;
}


TORCH_LIBRARY_FRAGMENT(ddt_inspector, m) {
  m.def("ddt_inspector::compute_consecutive_fopds", TORCH_FN(compute_consecutive_fopds));
  m.def("ddt_inspector::hist_mine_consecutive_fopds", TORCH_FN(hist_mine_consecutive_fopds));
}