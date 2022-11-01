#include <assert.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <torch/types.h>

using std::vector;


std::tuple<double, int64_t, int64_t>  run_length_stats_coo(const at::Tensor& mat) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse());
  TORCH_CHECK(mat.dim() == 2);
  TORCH_CHECK(mat.dense_dim() == 0,
              "sparse_sparse_matmul_cpu: scalar values expected, got ",
              mat.dense_dim(),
              "D values");

  auto mat_indices_ = mat._indices().contiguous();
  auto mat_values = mat._values().contiguous();

  at::Tensor row_indices = mat_indices_.select(0, 0);
  at::Tensor col_indices = mat_indices_.select(0, 1);

  std::cout << row_indices.size(0) << std::endl;
  std::cout << col_indices.size(0) << std::endl;

  for (int i = 0; i < row_indices.size(0) - 1; i++) {
    std::cout << row_indices[i+1] - row_indices[i] << " " << col_indices[i+1] - col_indices[i] << std::endl;
  }

  return {1.f, 1, 1};
}


template<typename T>
std::tuple<double, int64_t, int64_t> _run_length_stats_csr(T* row_ptrs, T* col_indices, int64_t rows, int64_t cols) {
  T max_run_length = 0;
  T min_run_length = cols + 1;
  long acc_run_length = 0;
  long cnt_run_length = 0;

  for (int i = 0; i < rows; i++) {
    T prev_col_loc = 0;
    for (auto p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
      T run_length = col_indices[p] - prev_col_loc;

      acc_run_length += run_length;
      cnt_run_length += 1;

      if (run_length < min_run_length) { min_run_length = run_length; }
      if (run_length > max_run_length) { max_run_length = run_length; }

      prev_col_loc = col_indices[p];
    }
  }

  double avg = double(acc_run_length) / cnt_run_length;
  return { avg, min_run_length, max_run_length };
}


std::tuple<double, int64_t, int64_t> run_length_stats_csr(const at::sparse_csr::SparseCsrTensor& mat) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
  TORCH_CHECK(mat.dim() == 2);

  auto nnz = mat._nnz();
  auto rows = mat.size(0);
  auto cols = mat.size(1);

  auto col_indices = mat.col_indices();
  auto row_ptrs = mat.crow_indices();

  std::tuple<double, int64_t, int64_t> ret;

  AT_DISPATCH_INTEGRAL_TYPES(col_indices.type(), "_run_length_stats_csr", ([&] {
     ret = _run_length_stats_csr(
       row_ptrs.data_ptr<scalar_t>(),
       col_indices.data_ptr<scalar_t>(),
        rows, cols
     );
  }));

  return ret;
}


template<typename T>
std::tuple<double, int64_t, int64_t> _stride_stats_csr(T* row_ptrs, T* col_indices, int64_t rows, int64_t cols) {
  T max_run_length = 0;
  T min_run_length = cols + 1;
  long acc_run_length = 0;
  long cnt_run_length = 0;

  for (int i = 0; i < rows; i++) {
    T prev_col_loc = 0;
    for (auto p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
      T run_length = col_indices[p] - prev_col_loc;

      acc_run_length += run_length;
      cnt_run_length += 1;

      if (run_length < min_run_length) { min_run_length = run_length; }
      if (run_length > max_run_length) { max_run_length = run_length; }

      prev_col_loc = col_indices[p];
    }
  }

  double avg = double(acc_run_length) / cnt_run_length;
  return { avg, min_run_length, max_run_length };
}

std::tuple<double, int64_t, int64_t> stride_stats_csr(const at::sparse_csr::SparseCsrTensor& mat) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
  TORCH_CHECK(mat.dim() == 2);

  auto nnz = mat._nnz();
  auto rows = mat.size(0);
  auto cols = mat.size(1);

  auto col_indices = mat.col_indices();
  auto row_ptrs = mat.crow_indices();

  std::tuple<double, int64_t, int64_t> ret;

  AT_DISPATCH_INTEGRAL_TYPES(col_indices.type(), "_stride_stats_csr", ([&] {
     ret = _run_length_stats_csr(
         row_ptrs.data_ptr<scalar_t>(),
         col_indices.data_ptr<scalar_t>(),
         rows, cols
     );
  }));

  return ret;
}

template<typename T>
std::tuple<double, int64_t, int64_t> _row_similarity_stats_csr(T* row_ptrs, T* col_indices, int64_t rows, int64_t cols) {
  T max_run_length = 0;
  T min_run_length = cols + 1;
  long acc_run_length = 0;
  long cnt_run_length = 0;

  for (int i = 0; i < rows; i++) {
    T prev_col_loc = 0;
    for (auto p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
      T run_length = col_indices[p] - prev_col_loc;

      acc_run_length += run_length;
      cnt_run_length += 1;

      if (run_length < min_run_length) { min_run_length = run_length; }
      if (run_length > max_run_length) { max_run_length = run_length; }

      prev_col_loc = col_indices[p];
    }
  }

  double avg = double(acc_run_length) / cnt_run_length;
  return { avg, min_run_length, max_run_length };
}

std::tuple<double, int64_t, int64_t> row_similarity_stats_csr(const at::sparse_csr::SparseCsrTensor& mat) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
  TORCH_CHECK(mat.dim() == 2);

  auto nnz = mat._nnz();
  auto rows = mat.size(0);
  auto cols = mat.size(1);

  auto col_indices = mat.col_indices();
  auto row_ptrs = mat.crow_indices();

  std::tuple<double, int64_t, int64_t> ret;

  AT_DISPATCH_INTEGRAL_TYPES(col_indices.type(), "_row_similarity_stats_csr", ([&] {
     ret = _run_length_stats_csr(
         row_ptrs.data_ptr<scalar_t>(),
         col_indices.data_ptr<scalar_t>(),
         rows, cols
     );
  }));

  return ret;
}

template<typename T>
std::tuple<int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor> _tile_stats_csr_not_binned(
    int64_t tile_rows, int64_t tile_cols,
    T* row_ptrs, T* col_indices, int64_t rows, int64_t cols) {

  int64_t num_tiles_rows = (rows + tile_rows - 1) / tile_rows;
  int64_t num_tiles_cols = (cols + tile_cols - 1) / tile_cols;

  auto nnz = vector<vector<int>>(num_tiles_rows, vector<int>(num_tiles_cols));
  auto non_empty_tiles_per_row_panel = std::vector<int64_t>(num_tiles_rows);
  auto non_empty_tiles_offsets = std::vector<int64_t>(num_tiles_rows + 1);
  non_empty_tiles_offsets[0] = 0;

  #pragma omp parallel for
  for (int ii = 0; ii < rows; ii += tile_rows) {
    for (int i = ii; i < std::min(ii + tile_rows, rows); i++) {
      int ti = i / tile_rows;
      for (int p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
        int j =  col_indices[p];
        int tj = col_indices[p] / tile_cols;
        int jj = tj * tile_cols;
        if (nnz[ti][tj] == 0) {
          non_empty_tiles_per_row_panel[ti] += 1;
        }
        nnz[ti][tj] += 1;
      }
    }
  }

  auto non_empty_tile_mapping = vector<vector<int64_t>>(
      num_tiles_rows, vector<int64_t>(num_tiles_cols, -1));

  auto active_rows = vector<vector<vector<bool>>>(num_tiles_rows);
  auto active_cols = vector<vector<vector<bool>>>(num_tiles_rows);
  auto nnz_non_empty = vector<vector<int64_t>>(num_tiles_rows);
  auto non_empty_tile_mapping_tj = vector<vector<int64_t>>(num_tiles_rows);

  // Allocate memory appropriately for the active_rows and active_cols vectors
  for (int i = 0; i < num_tiles_rows; i++) {
    active_rows[i] = vector<vector<bool>>(
        non_empty_tiles_per_row_panel[i], vector<bool>(tile_rows));
    active_cols[i] = vector<vector<bool>>(
        non_empty_tiles_per_row_panel[i], vector<bool>(tile_cols));
    nnz_non_empty[i] = vector<int64_t>(non_empty_tiles_per_row_panel[i]);
    non_empty_tile_mapping_tj[i] = vector<int64_t>(non_empty_tiles_per_row_panel[i]);
  }

  #pragma omp parallel for
  for (int ti = 0; ti < num_tiles_rows; ti++) {
    int curr_row_non_empty_tiles_offset = 0;
    for (int tj = 0; tj < num_tiles_cols; tj++) {
      if (nnz[ti][tj] > 0) {
        if (curr_row_non_empty_tiles_offset >= non_empty_tiles_per_row_panel[ti]) {
            std::cerr << "ERROR: curr_row_non_empty_tiles_offset: " << curr_row_non_empty_tiles_offset << std::endl;
            std::cerr << "non_empty_tiles_per_row_panel[ti]: " << non_empty_tiles_per_row_panel[ti] << std::endl;
            exit(-1);
        }

        nnz_non_empty[ti][curr_row_non_empty_tiles_offset] = nnz[ti][tj];
        non_empty_tile_mapping[ti][tj] = curr_row_non_empty_tiles_offset;
        non_empty_tile_mapping_tj[ti][curr_row_non_empty_tiles_offset] = tj;
        curr_row_non_empty_tiles_offset++;
      }
    }
  }

  int64_t total_non_empty_tiles = 0;
  for (int64_t i = 0; i < non_empty_tiles_per_row_panel.size(); i++) {
    non_empty_tiles_offsets[i + 1] = non_empty_tiles_offsets[i] + non_empty_tiles_per_row_panel[i];
    total_non_empty_tiles += non_empty_tiles_per_row_panel[i];
  }

  auto curr_non_empty_tiles = std::vector<int64_t>(num_tiles_rows);
  #pragma omp parallel for
  for (int64_t ii = 0; ii < rows; ii += tile_rows) {
    for (int64_t i = ii; i < std::min(ii + tile_rows, rows); i++) {
      int ti = i / tile_rows;
      for (int p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
        int j =  col_indices[p];
        int tj = col_indices[p] / tile_cols;
        int jj = tj * tile_cols;

        int non_empty_tiles_offset = non_empty_tile_mapping[ti][tj];
        if (non_empty_tiles_offset < 0) {
          std::cerr << "ERROR: non_empty_tile_mapping[" << ti << "][" << tj;
          std::cerr << "] = " << non_empty_tile_mapping[ti][tj] << std::endl;
          exit(-1);
        }

        active_rows[ti][non_empty_tiles_offset][i - ii] = 1;
        active_cols[ti][non_empty_tiles_offset][j - jj] = 1;
      }
    }
  }

  torch::Tensor active_cols_ret = at::zeros({total_non_empty_tiles}, at::kFloat);
  torch::Tensor active_rows_ret = at::zeros({total_non_empty_tiles}, at::kFloat);
  torch::Tensor nnz_ret = at::zeros({total_non_empty_tiles}, at::kFloat);

  auto active_cols_ret_data = active_cols_ret.data_ptr<float>();
  auto active_rows_ret_data = active_rows_ret.data_ptr<float>();
  auto nnz_ret_data = nnz_ret.data_ptr<float>();

  double density_accumulator = 0;

  #pragma omp parallel for reduction(+:density_accumulator)
  for (int64_t ti = 0; ti < num_tiles_rows; ti++) {
    for (int64_t p = 0; p < non_empty_tiles_per_row_panel[ti]; p++) {
      int active_col_count = 0, active_row_count = 0;

      for (const auto& is_col_active : active_cols[ti][p])
        active_col_count += is_col_active;
      for (const auto& is_row_active : active_rows[ti][p])
        active_row_count += is_row_active;

      int64_t tj = non_empty_tile_mapping_tj[ti][p];
      int64_t tile_rows_act = std::min(tile_rows, rows - ti * tile_rows);
      int64_t tile_cols_act = std::min(tile_cols, cols - tj * tile_cols);

      float pct_active_cols = float(active_col_count) / float(tile_cols_act);
      float pct_active_rows = float(active_row_count) / float(tile_rows_act);

      if (pct_active_cols > 1) {
        std::cerr << "pct_active_cols: " << pct_active_cols << std::endl;
        exit(-1);
      }

      if (pct_active_rows > 1) {
        std::cerr << "pct_active_rows: " << pct_active_rows << std::endl;
        exit(-1);
      }

      float density = float(nnz_non_empty[ti][p]) / float(tile_rows_act * tile_cols_act);
      active_cols_ret_data[p+non_empty_tiles_offsets[ti]] = pct_active_cols;
      active_rows_ret_data[p+non_empty_tiles_offsets[ti]] = pct_active_rows;
      nnz_ret_data[p+non_empty_tiles_offsets[ti]] = density;
    }
  }

  double total_tiles = num_tiles_cols * num_tiles_rows;
  return { total_tiles, total_tiles - total_non_empty_tiles,
          active_rows_ret, active_cols_ret, nnz_ret };
}

std::tuple<int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor> tile_stats_csr_not_binned(
    int64_t tile_rows, int64_t tile_cols,
    const at::sparse_csr::SparseCsrTensor& mat
) {
  TORCH_INTERNAL_ASSERT(mat.is_sparse_csr());
  TORCH_CHECK(mat.dim() == 2);

  auto nnz = mat._nnz();
  auto rows = mat.size(0);
  auto cols = mat.size(1);

  auto col_indices = mat.col_indices();
  auto row_ptrs = mat.crow_indices();

  std::tuple<int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor> ret;

  AT_DISPATCH_INTEGRAL_TYPES(
    col_indices.type(), "_tile_stats_csr_not_binned", ([&] {
      ret = _tile_stats_csr_not_binned(
          tile_rows, tile_cols,
          row_ptrs.data_ptr<scalar_t>(),
          col_indices.data_ptr<scalar_t>(),
          rows, cols
      );
    }));

  return ret;
}



TORCH_LIBRARY_FRAGMENT(spmm_benchmarks, m) {
  m.def("spmm_benchmarks::run_length_stats_coo", TORCH_FN(run_length_stats_coo));
  m.def("spmm_benchmarks::run_length_stats_csr", TORCH_FN(run_length_stats_csr));
  m.def("spmm_benchmarks::stride_stats_csr", TORCH_FN(stride_stats_csr));
  m.def("spmm_benchmarks::tile_stats_csr_not_binned", TORCH_FN(tile_stats_csr_not_binned));
}