
#ifndef ndarray_types_H
#define ndarray_types_H

#include "laruen/utils/range.h"
#include <cstdint>
#include <vector>

using namespace laruen::utils;

typedef float float32_t;
typedef double float64_t;

using Shape = std::vector<uint64_t>;
using Strides = std::vector<uint64_t>;
using NDIndex = std::vector<uint32_t>;
using SliceRanges = std::vector<Range<uint32_t>>;

std::ostream& operator<<(std::ostream &strm, const Shape &shape);
std::ostream& operator<<(std::ostream &strm, const SliceRanges &slice_ranges);

#include "laruen/ndarray/ndarray_types.tpp"
#endif