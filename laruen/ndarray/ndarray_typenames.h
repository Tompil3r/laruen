
#ifndef NDARRAY_TYPENAMES_H
#define NDARRAY_TYPENAMES_H

#include "laruen/utils/range.h"

#include <vector>
#include <cstdint>

typedef float float32_t;
typedef double float64_t;
typedef std::vector<uint32_t> Shape;
typedef std::vector<uint64_t> Strides;
typedef std::vector<uint32_t> NDIndex;
typedef std::vector<laruen::utils::Range<uint32_t>> SliceRanges;

#endif