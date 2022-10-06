//
// Created by lwilkinson on 10/5/22.
//

#define restrict __restrict__
#include "taco.h"

// bin/taco "C(i,k)=A(i,j)*B(j,k)" -f=C:dd -f=A:ds -f=B:dd -t=A:float -t=B:float -t=C:float -s="split(i,i0,i1,16)" -s="pos(j,jpos,A)" -s="split(jpos,jpos0,jpos1,16)" -s="reorder(i0,i1,jpos0,k,jpos1)" -s="parallelize(i0,CPUThread,NoRaces)" -s="parallelize(k,CPUVector,IgnoreRaces)" -write-source=taco_kernel_16.h
static int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    int C1_dimension = (int)(C->dimensions[0]);
    int C2_dimension = (int)(C->dimensions[1]);
    float* restrict C_vals = (float*)(C->vals);
    int A1_dimension = (int)(A->dimensions[0]);
    int* restrict A2_pos = (int*)(A->indices[1][0]);
    int* restrict A2_crd = (int*)(A->indices[1][1]);
    float* restrict A_vals = (float*)(A->vals);
    int B1_dimension = (int)(B->dimensions[0]);
    int B2_dimension = (int)(B->dimensions[1]);
    float* restrict B_vals = (float*)(B->vals);

#pragma omp parallel for schedule(static)
    for (int32_t pC = 0; pC < (C1_dimension * C2_dimension); pC++) {
        C_vals[pC] = 0.0;
    }

#pragma omp parallel for schedule(runtime)
    for (int32_t i0 = 0; i0 < ((A1_dimension + 15) / 16); i0++) {
        for (int32_t i1 = 0; i1 < 16; i1++) {
            int32_t i = i0 * 16 + i1;
            if (i >= A1_dimension)
                continue;

            for (int32_t jpos0 = A2_pos[i] / 16; jpos0 < ((A2_pos[(i + 1)] + 15) / 16); jpos0++) {
                if (jpos0 * 16 < A2_pos[i] || (jpos0 * 16 + 16) + ((jpos0 * 16 + 16) - jpos0 * 16) >= A2_pos[(i + 1)]) {
                    for (int32_t k = 0; k < B2_dimension; k++) {
                        int32_t kC = i * C2_dimension + k;
                        float tjpos1C_val = 0.0;
                        for (int32_t jpos1 = 0; jpos1 < 16; jpos1++) {
                            int32_t jposA = jpos0 * 16 + jpos1;
                            if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
                                continue;

                            int32_t j = A2_crd[jposA];
                            int32_t kB = j * B2_dimension + k;
                            tjpos1C_val += A_vals[jposA] * B_vals[kB];
                        }
                        C_vals[kC] = C_vals[kC] + tjpos1C_val;
                    }
                }
                else {
#pragma clang loop interleave(enable) vectorize(enable)
#pragma GCC ivdep
                    for (int32_t k = 0; k < B2_dimension; k++) {
                        int32_t kC = i * C2_dimension + k;
                        float tjpos1C_val0 = 0.0;
#pragma GCC ivdep
                        for (int32_t jpos1 = 0; jpos1 < 16; jpos1++) {
                            int32_t jposA = jpos0 * 16 + jpos1;
                            int32_t j = A2_crd[jposA];
                            int32_t kB = j * B2_dimension + k;
                            tjpos1C_val0 += A_vals[jposA] * B_vals[kB];
                        }
                        C_vals[kC] = C_vals[kC] + tjpos1C_val0;
                    }
                }
            }
        }
    }
    return 0;
}


int compute_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    return compute(C, A, B);
}