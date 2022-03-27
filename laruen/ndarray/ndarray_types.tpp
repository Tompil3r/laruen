
#include "laruen/ndarray/ndarray_types.h"
#include <ostream>

// operator<< for Shape<N> is used as operator<< for NDIndex<N>
std::ostream& operator<<(std::ostream &strm, const Shape &shape)
{
    uint8_t size = shape.size();
    strm << '(';

    for(uint8_t idx = 0;idx < size - 1;idx++)
    {
        strm << shape[idx];
        strm << ", ";
    }

    strm << shape[size - 1];
    strm << ')';

    return strm;
}

std::ostream& operator<<(std::ostream &strm, const Strides &strides)
{
    uint8_t size = strides.size();
    strm << '(';

    for(uint8_t idx = 0;idx < size - 1;idx++)
    {
        strm << strides[idx];
        strm << ", ";
    }

    strm << strides[size - 1];
    strm << ')';

    return strm;
}

std::ostream& operator<<(std::ostream &strm, const SliceRanges &slice_ranges)
{
    uint8_t size = slice_ranges.size();
    strm << '(';

    for(uint8_t idx = 0;idx < size - 1;idx++)
    {
        strm << slice_ranges[idx];
        strm << ", ";
    }

    strm << slice_ranges[size - 1];
    strm << ')';

    return strm;
}
