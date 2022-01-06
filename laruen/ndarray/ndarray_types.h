
#ifndef ndarray_types_H
#define ndarray_types_H

#include "laruen/utils/range.h"
#include "laruen/utils/array.h"
#include <cstdint>

using namespace laruen::utils;

typedef float float32_t;
typedef double float64_t;

template <uint64_t N> using Shape = Array<uint32_t, N>;
template <uint64_t N> using Strides = Array<uint64_t, N>;
template <uint64_t N> using NDIndex = Array<uint32_t, N>;
template <uint64_t N> using SliceRanges = Array<Range<uint32_t>, N>;

#endif