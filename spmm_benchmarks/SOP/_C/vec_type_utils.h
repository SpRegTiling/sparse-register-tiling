//
// Created by lwilkinson on 5/25/22.
//

#ifndef DNN_SPMM_BENCH_VEC_TYPE_UTILS_H
#define DNN_SPMM_BENCH_VEC_TYPE_UTILS_H

#include <vectorclass.h>

template<typename _Scalar, int vector_width>
struct Vec { using Type = Vec8f; using Scalar = _Scalar; };

template<> struct Vec<float,  128> { using Type = Vec4f;  using Scalar = float; };
template<> struct Vec<float,  256> { using Type = Vec8f;  using Scalar = float; };
template<> struct Vec<float,  512> { using Type = Vec16f; using Scalar = float; };
template<> struct Vec<double, 128> { using Type = Vec2d;  using Scalar = double; };
template<> struct Vec<double, 256> { using Type = Vec4d;  using Scalar = double; };
template<> struct Vec<double, 512> { using Type = Vec8d;  using Scalar = double; };


#endif //DNN_SPMM_BENCH_VEC_TYPE_UTILS_H
