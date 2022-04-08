
#include "src/ndarray/ndarray_types.h"
#include "src/ndarray/ndarray.h"
#include <string>
#include <tuple>
#include <cstdint>
#include <type_traits>

using laruen::ndarray::NDArray;

// operator<< for Shape is used as operator<< for NDIndex and Strides 
std::string str(const Shape &shape) {
    std::string str;

    uint8_t size = shape.size();
    str.push_back('(');

    for(uint8_t i = 0;i < size - 1;i++) {
        str += std::to_string(shape[i]);
        str.push_back(',');
        str.push_back(' ');
    }

    str += std::to_string(shape[size - 1]);
    str.push_back(')');

    return str;
}

std::string str(const SliceRanges &slice_ranges) {
    std::string str;

    uint8_t size = slice_ranges.size();
    str.push_back('(');

    for(uint8_t i = 0;i < size - 1;i++) {
        str += slice_ranges[i].str();
        str.push_back(',');
        str.push_back(' ');
    }

    str += slice_ranges[size - 1].str();
    str.push_back(')');

    return str;
}

// ** experimental **
namespace types {
    template <> struct next_signed<int8_t> { typedef int16_t type; };
    template <> struct next_signed<uint8_t> { typedef int16_t type; };
    template <> struct next_signed<int16_t> { typedef int32_t type; };
    template <> struct next_signed<uint16_t> { typedef int32_t type; };
    template <> struct next_signed<int32_t> { typedef int64_t type; };
    template <> struct next_signed<uint32_t> { typedef int64_t type; };
    template <> struct next_signed<int64_t> { typedef float64_t type; };
    template <> struct next_signed<uint64_t> { typedef float64_t type; };
    template <> struct next_signed<float32_t> { typedef float64_t type; };
    template <> struct next_signed<float64_t> { typedef float64_t type; };

    template <typename T, typename T2>
    struct max_type {
        typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<T2>,
            std::conditional_t<sizeof(T) >= sizeof(T2), T, T2>,
            std::conditional_t<std::is_floating_point_v<T>, T, T2>> type;
    };

    template <typename T, typename T2> struct float_type {
        typedef std::conditional_t<std::is_floating_point_v<T>, T, T2> type;
    };

    template <typename T, typename T2> struct integer_type {
        typedef std::conditional_t<std::is_integral_v<T>, T, T2> type;
    };

    template <typename T, typename T2>
    struct combine_types {
    /*
        code simplification:
        if(is_int(T) == is_int(T2)) {
            if(is_signed(T) == is_signed(T2) || (is_signed(max_type(T, T2)) && sizeof(T) != sizeof(T2))) {
                type = max_type(T, T2);
            }
            else {
                type = next_signed(max(T, T2));
            }
        }
        else {
            if(sizeof(int_type(T, T2)) >= sizeof(float_type(T, T2))) {
                type = next_signed(max_type(T, T2));
            }   
            else {
                type = max_type(T, T2)
            }
        }
    */
    typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<T2>,
        // group a - both ints or both floats
        std::conditional_t<std::is_signed_v<T> == std::is_signed_v<T2> ||
            (std::is_signed_v<max_type_t<T, T2>> && sizeof(T) != sizeof(T2)),
            // sub group a1 - (both signed or both unsigned) or (one signed and one unsigned and have different sizes)
            max_type_t<T, T2>,
            // sub group a2 - one signed and one unsigned (order does not matter) and max or equal size is unsigned
            next_signed_t<max_type_t<T, T2>>>,

        // group b - one int and one float (order does not matter)
        std::conditional_t<sizeof(integer_type_t<T, T2>) >= sizeof(float_type_t<T, T2>),
            // sub group b1 - the size of the integer type is bigger or equal to the size of the float type
            next_signed_t<max_type_t<T, T2>>,
            // sub group b2 - the size of the integer type is smaller than the size of the float type
            max_type_t<T, T2>>> type;
    };
  
    template <typename T> struct is_ndarray {
        static constexpr bool value = false;
    };

    template <typename T> struct is_ndarray<NDArray<T>> {
        static constexpr bool value = true;
    };
}
