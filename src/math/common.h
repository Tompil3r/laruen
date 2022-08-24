
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

    template <typename T, typename TT>
    inline constexpr auto remainder(T lhs, TT rhs) noexcept {
        if constexpr(std::is_arithmetic_v<T> && std::is_arithmetic_v<TT>) {
            if constexpr(std::is_same_v<T, long double> || std::is_same_v<TT, long double>) {
                return fmodl(lhs, rhs);
            }
            else if constexpr(std::is_same_v<T, double> || std::is_same_v<TT, double>) {
                return fmod(lhs, rhs);
            }
            else if constexpr(std::is_same_v<T, float> || std::is_same_v<TT, float>) {
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

    template <typename T, typename TT>
    inline constexpr T pow(T base, TT exp) {
        if constexpr(std::is_floating_point_v<TT>) {
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

    template <typename T, typename TT>
    inline constexpr auto log(T base, TT power) noexcept {
        return std::log10(power) / std::log10(base);
    }

    template <typename T>
    inline constexpr T sign(T num) noexcept {
        return (num > 0) - (num < 0);
    }

};

#endif