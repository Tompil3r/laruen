
#ifndef NDLIB_TYPES_H_
#define NDLIB_TYPES_H_

#include <cstdint>
#include <vector>
#include "src/ndlib/range.h"

namespace laruen::ndlib {
    using float32_t = std::float_t;
    using float64_t = std::double_t;

    using Shape = std::vector<uint_fast64_t>;
    using Strides = std::vector<uint_fast64_t>;
    using NDIndex = std::vector<uint_fast64_t>;
    using SliceRanges = std::vector<Range<uint_fast64_t>>;
    using Axes = std::vector<uint_fast8_t>;
}

#endif