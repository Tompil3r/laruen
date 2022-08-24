
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <limits>

namespace laruen::math::utils {
    template <typename T, typename TT>
    constexpr T ipow(T base, TT exp) noexcept {
        T result = 1;

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

    template <typename T>
    constexpr inline T nonzero(T num) noexcept {
        return num ? num : std::numeric_limits<T>::min();
    }
}

#endif