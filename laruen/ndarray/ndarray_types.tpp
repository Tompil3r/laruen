
#include "laruen/ndarray/ndarray_types.h"
#include <ostream>

// operator<< for Shape is used as operator<< for NDIndex and Strides 
std::ostream& operator<<(std::ostream &strm, const Shape &shape) {
    uint8_t size = shape.size();
    strm << '(';

    for(uint8_t i = 0;i < size - 1;i++) {
        strm << shape[i];
        strm << ", ";
    }

    strm << shape[size - 1];
    strm << ')';

    return strm;
}

std::ostream& operator<<(std::ostream &strm, const SliceRanges &slice_ranges) {
    uint8_t size = slice_ranges.size();
    strm << '(';

    for(uint8_t i = 0;i < size - 1;i++) {
        strm << slice_ranges[i];
        strm << ", ";
    }

    strm << slice_ranges[size - 1];
    strm << ')';

    return strm;
}
