//
// Created by lwilkinson on 6/5/22.
//

#ifndef DNN_SPMM_BENCH_ASPT_H
#define DNN_SPMM_BENCH_ASPT_H

template<typename Scalar>
struct InspectorMetadata {
    int npanel;
    int nr;

    int *mcsr_e; // can be short type
    int *mcsr_cnt;
    //int *mcsr_list;
    int *mcsr_chk;

    int* row_ptrs_padded;
    int* col_indices_reordered;
    Scalar* values_reordered;

    double avg, vari;
    int nThread;

    int *special;
    int *special2;
    int special_p;
};



InspectorMetadata<float> inspect(
    int nr0, int nc, int ne,
    int* row_ptrs, int* col_indices, float* values,
    int NTHREAD,
    int ASpT_block_height = 128
);

void execute(
    const InspectorMetadata<float>& meta,
    int nr0, int nc, int sc,
    int* row_ptrs, int* col_indices, float* values,
    float* vin,
    float* vout,
    int ASpT_block_height = 128
);

void free(InspectorMetadata<float> &meta);

#endif //DNN_SPMM_BENCH_ASPT_H
