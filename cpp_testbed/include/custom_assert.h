//
// Created by lwilkinson on 6/1/22.
//

#ifndef DNN_SPMM_BENCH_CUSTOM_ASSERT_H
#define DNN_SPMM_BENCH_CUSTOM_ASSERT_H

#define ASSERT_RELEASE(cond) if (!(cond)) { fprintf(stderr, "Assertion: " #cond  " %s:line %d: \n", __FILE__,__LINE__);  exit(-1); }

#endif //DNN_SPMM_BENCH_CUSTOM_ASSERT_H
