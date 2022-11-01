#include <assert.h>
#include <ATen/ATen.h>
#include <torch/types.h>

std::tuple<int64_t, int64_t, at::Tensor, at::Tensor> load_smtx(const std::string &path) {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) {
    std::cout << "File could not be found..." << std::endl;
    exit(1);
  }

  std::string line;

  std::getline(file, line);
  std::replace(line.begin(), line.end(), ',', ' ');
  std::istringstream first_line(line);

  int rows, cols, nnz;
  first_line >> rows;
  first_line >> cols;
  first_line >> nnz;

  auto row_ptrs = at::empty({rows + 1}, at::kInt);
  auto col_indices = at::empty({nnz}, at::kInt);

  for (int i = 0; i < rows + 1; i++) {
    int tmp;
    file >> tmp;
    row_ptrs[i] = tmp;
  }

  // Go to next line
  char next;
  while (file.get(next)) { if (next == '\n') break; }

  // Read in col_indices
  for (int i = 0; i < nnz; i++) {
    int tmp;
    file >> tmp;
    col_indices[i] = tmp;
   }

  return {rows, cols, row_ptrs, col_indices};
}

TORCH_LIBRARY_FRAGMENT(spmm_benchmarks, m) {
  m.def("spmm_benchmarks::load_smtx", TORCH_FN(load_smtx));
}