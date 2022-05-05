
#ifndef MATH_COMMON_H
#define MATH_COMMON_H

#include "src/ndlib/ndarray_types.h"
#include <type_traits>
#include <cmath>

namespace laruen::math::common {
    template <typename T>
    inline constexpr T max(T value1, T value2) noexcept {
        return (value1 > value2 ? value1 : value2);
    }

    template <typename T>
    inline constexpr T min(T value1, T value2) noexcept {
        return (value1 < value2 ? value1 : value2);
    }

    template <typename T1, typename T2>
    inline constexpr auto remainder(T1 lhs, T2 rhs) noexcept {
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

    template <typename T1, typename T2>
    constexpr T1 ipow(T1 base, T2 exp) noexcept {
        T1 result = 1;

        for(;;) {
            if(exp & 1) {
                result *= base;
            }
            exp >>= 1;
            if(!exp) {
                break;
            }
            base *= base;
        }

        return result;
    }

    template <typename T1, typename T2>
    inline constexpr T1 pow(T1 base, T2 exp) {
        if constexpr(std::is_floating_point_v<T2>) {
            return std::pow(base, exp);
        }
        else {
            return ipow(base, exp);
        }
    }
};

#endif