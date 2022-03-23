
#ifndef ndarray_types_H
#define ndarray_types_H

#include "laruen/utils/range.h"
#include "laruen/utils/container.h"
#include <cstdint>

using namespace laruen::utils;

typedef float float32_t;
typedef double float64_t;

template <uint8_t N> using Shape = Container<uint32_t, uint8_t, N>;
template <uint8_t N> using Strides = Container<uint64_t, uint8_t ,N>;
template <uint8_t N> using NDIndex = Container<uint32_t, uint8_t ,N>;
template <uint8_t N> using SliceRanges = Container<Range<uint32_t>, uint8_t, N>;

template <uint8_t N> std::ostream& operator<<(std::ostream &strm, const Shape<N> &shape);
template <uint8_t N> std::ostream& operator<<(std::ostream &strm, const Strides<N> &strides);
template <uint8_t N> std::ostream& operator<<(std::ostream &strm, const SliceRanges<N> &slice_ranges);

#include "laruen/ndarray/ndarray_types.tpp"
#endif