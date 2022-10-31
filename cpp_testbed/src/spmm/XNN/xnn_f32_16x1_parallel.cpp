//
// Created by Kazem on 10/19/22.
//

#include <assert.h>
#include <stddef.h>
#include <arm_neon.h>

#include "SpMM_XNN.h"

//
// From: https://github.com/google/XNNPACK/blob/295ea1aaf511ee594f3500da8465d20ea024fec1/src/xnnpack/common.h
//      Removed minmax for fairness
//      This file is part of XNNPACK, a library for efficient neural network inference.
//

#if defined(__GNUC__)
#define XNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
#define XNN_LIKELY(condition) (!!(condition))
#define XNN_UNLIKELY(condition) (!!(condition))
#endif

#define restrict __restrict


//#define MIN_MAX

size_t divide_round_up(size_t n, size_t q) {
return (n % q == 0) ? n / q : n / q + 1;
}

void xnn_f32_spmm_minmax_ukernel_16x1__neon_parallel(
  size_t input_size, // ncols
  size_t nc, // nrows
  const float*restrict input, // B
  const float*restrict weights, // A
  const int32_t*restrict widx_dmap,
  const uint32_t*restrict nidx_nnzmap,
  float*restrict output, // C
  size_t output_stride,
  size_t num_threads
)
{
    size_t mc = input_size;
 if (num_threads > 1) {
  const size_t mr = mc;
  const size_t target_tiles_per_thread = 5;
  const size_t max_mc = divide_round_up(mc, num_threads * target_tiles_per_thread);
  if (max_mc < mc) {
   mc = std::min(mc, divide_round_up(mc, max_mc * mr) * mr);
  }
 }
#pragma omp parallel for num_threads(num_threads)
 for(int i = 0; i<input_size; i+=mc){
  size_t offset =  i;
  const float*restrict input_part = input + offset;
  float*restrict output_part = output + offset;
  xnn_f32_spmm_minmax_ukernel_16x1__neon(mc, nc, input_part, weights, widx_dmap,
                                         nidx_nnzmap, output_part,
                                         output_stride);
 }
}
