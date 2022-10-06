// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
// bin/taco "C(i,k)=A(i,j)*B(j,k)" -f=C:dd -f=A:ds -f=B:dd -t=A:float -t=B:float -t=C:float -s="split(i,i0,i1,16)" -s="pos(j,jpos,A)" -s="split(jpos,jpos0,jpos1,4)" -s="reorder(i0,i1,jpos0,k,jpos1)" -s="parallelize(i0,CPUThread,NoRaces)" -s="parallelize(k,CPUVector,IgnoreRaces)" -write-source=taco_kernel_4.h

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#if _OPENMP
#include <omp.h>
#endif
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  uint8_t*     fill_value;    // tensor fill value
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#if !_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
#endif


taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types);
void deinit_taco_tensor_t(taco_tensor_t* t);