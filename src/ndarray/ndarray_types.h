
#ifndef ndarray_types_H
#define ndarray_types_H

#include "src/utils/range.h"
#include <cstdint>
#include <vector>
#include <tuple>
#include <string>

using namespace laruen::utils;

typedef float float32_t;
typedef double float64_t;

using Shape = std::vector<uint64_t>;
using Strides = std::vector<uint64_t>;
using NDIndex = std::vector<uint64_t>;
using SliceRanges = std::vector<Range<uint64_t>>;

std::string str(const Shape &shape);
std::string str(const SliceRanges &slice_ranges);

// ** experimental **
namespace types {
    template <typename T, bool B> struct next_signed; // B - next only (do not include the current type if signed)
    template <typename T, typename T2, typename = void> struct max_type;
    template <typename T, typename T2> constexpr bool type_contained();
}

#include "src/ndarray/ndarray_types.tpp"
#endif