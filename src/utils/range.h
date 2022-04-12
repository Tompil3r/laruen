
#ifndef RANGE_H
#define RANGE_H

#include <string>

namespace laruen::utils {
    
    template <typename T> struct Range {
        T start;
        T end;
        T step;

        constexpr Range(T end);
        constexpr Range(T start, T end);
        constexpr Range(T start, T end, T step);
        std::string str() const;
    };
};

#include "src/utils/range.tpp"
#endif