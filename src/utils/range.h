
#ifndef RANGE_H
#define RANGE_H

#include <string>

namespace laruen::utils {
    
    template <typename T> struct Range {
        T start;
        T end;
        T step;

        constexpr Range(T start=0, T end=0, T step=1);
        std::string str() const;
    };
};

#include "src/utils/range.tpp"
#endif