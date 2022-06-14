
#ifndef NDLIB_TYPES_H_
#define NDLIB_TYPES_H_

#include "src/ndlib/range.h"
#include <cstdint>
#include <vector>
#include <tuple>
#include <string>
#include <cmath>

typedef std::float_t float32_t;
typedef std::double_t float64_t;

using Shape = std::vector<uint_fast64_t>;
using Strides = std::vector<uint_fast64_t>;
using NDIndex = std::vector<uint_fast64_t>;
using SliceRanges = std::vector<laruen::ndlib::Range<uint_fast64_t>>;
using Axes = std::vector<uint_fast8_t>;

std::string str(const Shape &shape) noexcept;
std::string str(const SliceRanges &slice_ranges) noexcept;

// ** experimental **
namespace types {
    template <typename T> struct next_signed;
    template <typename T, typename T2> struct max_type;
    template <typename T, typename T2> struct float_type;
    template <typename T, typename T2> struct integer_type;
    template <typename T, typename T2> struct combine_types;
    template <typename T> struct is_ndarray;

    template <typename T> using next_signed_t = typename next_signed<T>::type;
    template <typename T, typename T2> using max_type_t = typename max_type<T, T2>::type;
    template <typename T, typename T2> using float_type_t = typename float_type<T, T2>::type;
    template <typename T, typename T2> using integer_type_t = typename integer_type<T, T2>::type;
    template <typename T, typename T2> using combine_types_t = typename combine_types<T, T2>::type;

    template <typename T> inline constexpr bool is_ndarray_v = is_ndarray<T>::value;
    template <typename T, typename T2> inline constexpr bool both_integers_v = (std::is_integral_v<T> && std::is_integral_v<T2>);
    template <typename T, typename T2> inline constexpr bool atleast_one_float_v = (std::is_floating_point_v<T> || std::is_floating_point_v<T2>);
}

#include "src/ndlib/ndlib_types.tpp"
#endif