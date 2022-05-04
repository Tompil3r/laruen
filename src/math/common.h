
#ifndef MATH_COMMON_H
#define MATH_COMMON_H

#include "src/ndlib/ndarray_types.h"
#include <type_traits>
#include <cmath>

namespace laruen::math::common {
    template <typename T> inline constexpr T max(T value1, T value2) noexcept {
        return (value1 > value2 ? value1 : value2);
    }

    template <typename T> inline constexpr T min(T value1, T value2) noexcept {
        return (value1 < value2 ? value1 : value2);
    }

    template <typename T1, typename T2> inline constexpr auto remainder(T1 lhs, T2 rhs) noexcept {
        if constexpr(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>) {
            if constexpr(std::is_same_v<types::max_type_t<T1, T2>, float>) {
                return fmodf(lhs, rhs);
            }
            else if constexpr(std::is_same_v<types::max_type_t<T1, T2>, double>) {
                return fmod(lhs, rhs);
            }
            else if constexpr(std::is_same_v<types::max_type_t<T1, T2>, long double>) {
                return fmodl(lhs, rhs);
            }
            else {
                return lhs % rhs;
            }
        }
        else {
            return lhs % rhs;
        }
    }
};

#endif