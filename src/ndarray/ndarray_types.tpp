
#include "src/ndarray/ndarray_types.h"
#include <string>
#include <tuple>
#include <cstdint>
#include <type_traits>

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
    template <> struct next_signed<uint8_t> { typedef int16_t type; };
    template <> struct next_signed<uint16_t> { typedef int32_t type; };
    template <> struct next_signed<uint32_t> { typedef int64_t type; };
    template <> struct next_signed<uint64_t> { typedef float32_t type; };
    template <> struct next_unsigned<int8_t> { typedef uint16_t type; };
    template <> struct next_unsigned<int16_t> { typedef uint32_t type; };
    template <> struct next_unsigned<int32_t> { typedef uint64_t type; };
    template <> struct next_unsigned<int64_t> { typedef float32_t type; };

    template <typename T, typename T2>
    constexpr bool type_contained() {
        return sizeof(T) >= sizeof(T2) && (std::is_floating_point<T>::value || (std::is_signed<T>::value
        || std::is_signed<T>::value == std::is_signed<T2>::value));
    }
}
