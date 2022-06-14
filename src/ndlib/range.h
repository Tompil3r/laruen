
#ifndef NDLIB_RANGE_H_
#define NDLIB_RANGE_H_

#include <string>

namespace laruen::ndlib {
    
    template <typename T> struct Range {
        T start;
        T end;
        T step;

        constexpr Range(T end) noexcept;
        constexpr Range(T start, T end) noexcept;
        constexpr Range(T start, T end, T step) noexcept;
        std::string str() const noexcept;
    };
};

#include "src/ndlib/range.tpp"
#endif