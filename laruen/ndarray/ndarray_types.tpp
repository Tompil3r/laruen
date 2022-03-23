
#include "laruen/ndarray/ndarray_types.h"
#include <ostream>

// operator<< for Shape<N> is used as operator<< for NDIndex<N>
template <uint8_t N>
std::ostream& operator<<(std::ostream &strm, const Shape<N> &shape)
{
    strm << '(';

    for(uint8_t idx = 0;idx < N - 1;idx++)
    {
        strm << shape[idx];
        strm << ", ";
    }

    strm << shape[N - 1];
    strm << ')';

    return strm;
}

template <uint8_t N>
std::ostream& operator<<(std::ostream &strm, const Strides<N> &strides)
{
    strm << '(';

    for(uint8_t idx = 0;idx < N - 1;idx++)
    {
        strm << strides[idx];
        strm << ", ";
    }

    strm << strides[N - 1];
    strm << ')';

    return strm;
}

template <uint8_t N>
std::ostream& operator<<(std::ostream &strm, const SliceRanges<N> &slice_ranges)
{
    strm << '(';

    for(uint8_t idx = 0;idx < N - 1;idx++)
    {
        strm << slice_ranges[idx];
        strm << ", ";
    }

    strm << slice_ranges[N - 1];
    strm << ')';

    return strm;
}
