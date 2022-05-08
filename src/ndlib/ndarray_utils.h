
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndlib/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>

namespace laruen::ndlib { template <typename T, bool C> class NDArray; }
using laruen::ndlib::NDArray;

namespace laruen::ndlib::utils {
    template <bool = false>
    uint_fast8_t rev_count_diff(const Shape&, const Shape&) noexcept;

    template <bool = false>
    Shape broadcast(const Shape&, const Shape&);

    template <typename T, bool C, typename T2, bool C2>
    NDArray<T, false> broadcast_reorder(NDArray<T, C>&, const NDArray<T2, C2>&);

    inline uint_fast64_t ceil_index(float64_t index) noexcept {
        return (uint_fast64_t)index + ((uint_fast64_t)index < index);
    }
}

namespace laruen::ndlib::utils::operations {
    template <typename T1, typename T2>
    inline constexpr T1& addition(T1 &lhs, const T2 &rhs) noexcept {
        return lhs += rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR addition(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs + rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& subtraction(T1 &lhs, const T2 &rhs) noexcept {
        return lhs -= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR subtraction(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs - rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& multiplication(T1 &lhs, const T2 &rhs) noexcept {
        return lhs *= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR multiplication(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs * rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& division(T1 &lhs, const T2 &rhs) noexcept {
        return lhs /= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR division(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs / rhs;
    }
    
    template <typename T1, typename T2>
    inline constexpr T1& bit_xor(T1 &lhs, const T2 &rhs) noexcept {
        return lhs ^= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR bit_xor(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs ^ rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& bit_and(T1 &lhs, const T2 &rhs) noexcept {
        return lhs &= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR bit_and(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs & rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& bit_or(T1 &lhs, const T2 &rhs) noexcept {
        return lhs |= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR bit_or(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs | rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& bit_shl(T1 &lhs, const T2 &rhs) noexcept {
        return lhs <<= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR bit_shl(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs << rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& bit_shr(T1 &lhs, const T2 &rhs) noexcept {
        return lhs >>= rhs;
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR bit_shr(const T1 &lhs, const T2 &rhs) noexcept {
        return lhs >> rhs;
    }

    template <typename T1, typename T2>
    inline constexpr T1& remainder(T1 &lhs, const T2 &rhs) noexcept {
        return lhs = math::common::remainder(lhs, rhs);
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR remainder(const T1 &lhs, const T2 &rhs) noexcept {
        return math::common::remainder(lhs, rhs);
    }

    template <typename T1, typename T2>
    inline constexpr T1& power(T1 &lhs, const T2 &rhs) noexcept {
        return lhs = math::common::pow(lhs, rhs);
    }
    
    template <typename TR, typename T1, typename T2>
    inline constexpr TR power(const T1 &lhs, const T2 &rhs) noexcept {
        return math::common::pow(lhs, rhs);
    }
}


#include "src/ndlib/ndarray_utils.tpp"
#endif