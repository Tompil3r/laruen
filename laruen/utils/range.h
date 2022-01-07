
#ifndef RANGE_H
#define RANGE_H

#include <ostream>

namespace laruen::utils
{
    template <typename T> struct Range
    {
        T start;
        T end;
        T step;

        Range(T start=0, T end=0, T step=1);
    };
};

#include "laruen/utils/range.tpp"
#endif