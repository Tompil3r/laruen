
#ifndef MATH_COMMON_H
#define MATH_COMMON_H

namespace laruen::math::common {
    template <typename T> constexpr T max(T value1, T value2) {
        return (value1 > value2 ? value1 : value2);
    }

    template <typename T> constexpr T min(T value1, T value2) {
        return (value1 < value2 ? value1 : value2);
    }
};

#endif