
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

namespace laruen::math::utils {
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
}

#endif