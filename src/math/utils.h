
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

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
}

#endif