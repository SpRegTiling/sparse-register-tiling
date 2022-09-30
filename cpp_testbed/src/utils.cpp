//
// Created by Kazem on 10/11/19.
//

#include <sparse_io.h>
#include <cmath>
#include <utils.h>
namespace sym_lib {

bool time_cmp(timing_measurement a, timing_measurement b) {
  return a.elapsed_time < b.elapsed_time;
}

timing_measurement time_median(std::vector<timing_measurement> time_array) {
  size_t n = time_array.size();
  if (n == 0) {
    timing_measurement t;
    t.elapsed_time = -1;
    return t;
  }
  std::sort(time_array.begin(), time_array.end(), time_cmp);
  if (n == 1)
    return time_array[0];
  return time_array[n / 2];
}

/* Safely compute a*k, where k should be small, and check for integer overflow.
 * If overflow occurs, return 0 and set OK to FALSE.  Also return 0 if OK is
 * FALSE on input. */
size_t mult_size_t(size_t a, size_t k, int* ok) {
  size_t p = 0, s;
  while (*ok) {
    if (k % 2) {
      p = p + a;
      (*ok) = (*ok) && (p >= a);
    }
    k = k / 2;
    if (!k)
      return (p);
    s = a + a;
    (*ok) = (*ok) && (s >= a);
    a = s;
  }
  return (0);
}

size_t add_size_t(size_t a, size_t b, int* ok) {
  size_t s = a + b;
  (*ok) = (*ok) && (s >= a);
  return ((*ok) ? s : 0);
}

int* partition_by_weight(
    int n,
    const int* set,
    const double* weight,
    int n_parts,
    double* target_weight) {
  double* even_weight;
  if (target_weight)
    even_weight = target_weight;
  else {
    even_weight = new double[n_parts];
    double even_w = std::ceil(sum_vector(n, weight) / n_parts);
    std::fill_n(even_weight, n_parts, even_w);
  }
  int* indices = new int[n_parts + 1]();
  int j = 0;
  for (int i = 0; i < n_parts; ++i) {
    double c_wgt = 0;
    while (c_wgt < even_weight[i] && j < n) {
      int c_n = set[j];
      c_wgt += weight[c_n];
      j++;
    }
    indices[i + 1] = j;
  }
  if (!target_weight)
    delete[] even_weight;
  return indices;
}

}