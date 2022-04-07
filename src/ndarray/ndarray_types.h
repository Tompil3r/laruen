
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
    template <typename T> struct next_signed;
    template <typename T, typename T2> struct max_type;
    template <typename T, typename T2> struct float_type;
    template <typename T, typename T2> struct integer_type;
    template <typename T, typename T2> struct combine_types;

    template <typename T> using next_signed_t = typename next_signed<T>::type;
    template <typename T, typename T2> using max_type_t = typename max_type<T, T2>::type;
    template <typename T, typename T2> using float_type_t = typename float_type<T, T2>::type;
    template <typename T, typename T2> using integer_type_t = typename integer_type<T, T2>::type;
    template <typename T, typename T2> using combine_types_t = typename combine_types<T, T2>::type;
}

#include "src/ndarray/ndarray_types.tpp"
#endif