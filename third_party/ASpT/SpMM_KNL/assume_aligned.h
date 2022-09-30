//
// Created by lwilkinson on 6/5/22.
//

#ifndef DNN_SPMM_BENCH_ASSUME_ALIGNED_H
#define DNN_SPMM_BENCH_ASSUME_ALIGNED_H

#include <cstddef>
#include <cstdint>

template <std::size_t N, typename T>
#if defined(__clang__) || defined(__GNUC__)
__attribute__((always_inline))
#elif defined(_MSC_VER)
__forceinline
#endif
#if (defined(__GNUC__) && !defined(__ICC)) && !defined(__clang__)
[[nodiscard]]
#endif
constexpr T* assume_aligned(T* ptr) noexcept
{
#if defined(__clang__) || (defined(__GNUC__) && !defined(__ICC))
    return reinterpret_cast<T*>(__builtin_assume_aligned(ptr, N));
#elif defined(_MSC_VER)
    if ((reinterpret_cast<std::uintptr_t>(ptr) & ((1 << N) - 1)) == 0)
        return ptr;
    else
        __assume(0);
#elif defined(__ICC)
    switch (N) {
        case 2: __assume_aligned(ptr, 2); break;
        case 4: __assume_aligned(ptr, 4); break;
        case 8: __assume_aligned(ptr, 8); break;
        case 16: __assume_aligned(ptr, 16); break;
        case 32: __assume_aligned(ptr, 32); break;
        case 64: __assume_aligned(ptr, 64); break;
        case 128: __assume_aligned(ptr, 128); break;
    }
    return ptr;
#else
    // Unknown compiler — do nothing
    return ptr;
#endif
}

#endif //DNN_SPMM_BENCH_ASSUME_ALIGNED_H
