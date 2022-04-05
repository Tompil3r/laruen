
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
    /*
        type id range: 1 - 10
        1 - 8 = signed, unsigned... (min size to max)
        9 - 10 = float32_t, float64_t
    */

    template <typename T, typename Tuple> struct Type;
    template <typename T, typename... Types> struct Type<T, std::tuple<T, Types...>>;
    template <typename T, typename U, typename... Types> struct Type<T, std::tuple<U, Types...>>;

    template <typename T> constexpr uint64_t type_id();
    constexpr bool type_signed(uint64_t type_id);
    constexpr bool type_decimal(uint64_t type_id);
    template <typename T, typename U> constexpr bool type_contained();
}

#include "src/ndarray/ndarray_types.tpp"
#endif