
#ifndef MATH_COMMON_H_
#define MATH_COMMON_H_

#include <type_traits>
#include <cmath>
#include "src/math/utils.h"

namespace laruen::math::common {
    template <typename T>
    inline constexpr T max(T value1, T value2) noexcept {
        return (value1 > value2 ? value1 : value2);
    }

    template <typename T>
    inline constexpr T min(T value1, T value2) noexcept {
        return (value1 < value2 ? value1 : value2);
    }

    template <typename T>
    inline constexpr T abs(T value) noexcept {
        return value >= 0 ? value : -value;
    }

    template <typename T1, typename T2>
    inline constexpr auto remainder(T1 lhs, T2 rhs) noexcept {
        if constexpr(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>) {
            if constexpr(std::is_same_v<T1, long double> || std::is_same_v<T2, long double>) {
                return fmodl(lhs, rhs);
            }
            else if constexpr(std::is_same_v<T1, double> || std::is_same_v<T2, double>) {
                return fmod(lhs, rhs);
            }
            else if constexpr(std::is_same_v<T1, float> || std::is_same_v<T2, float>) {
                return fmodf(lhs, rhs);
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
    inline constexpr T1 pow(T1 base, T2 exp) {
        if constexpr(std::is_floating_point_v<T2>) {
            return std::pow(base, exp);
        }
        else {
            return laruen::math::utils::ipow(base, exp);
        }
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    inline bool is_pow2(T n) noexcept {
        return n && !(n & (n - 1));
    }
};

#endif