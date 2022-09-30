//
// Created by lwilkinson on 5/10/22.
//

#ifndef DNN_SPMM_BENCH_TEMPLATE_UTILS_H
#define DNN_SPMM_BENCH_TEMPLATE_UTILS_H

#include <tuple>

#define SUPPORTED_SCALARS float,double

// NOTE Does not support namespaces
#define INSTANTIATE_FOR_SUPPORTED_SCALARS(f)               \
  template<typename... Ts>                                 \
  static constexpr auto instantiate_##f () {                  \
    return std::tuple_cat(std::make_tuple(f<Ts>)...); }    \
  static constexpr auto _##f##_templates __attribute__((used)) \
      = instantiate_##f <SUPPORTED_SCALARS>();






#endif // DNN_SPMM_BENCH_TEMPLATE_UTILS_H
