
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
    template <typename T, typename T2> struct max_type { typedef typename std::conditional<sizeof(T) >= sizeof(T2), T, T2> type; };
    
    template <typename T, typename T2>
    constexpr bool type_contained() {
        return sizeof(T) >= sizeof(T2) && (std::is_floating_point<T>::value || (std::is_signed<T>::value
        || std::is_signed<T>::value == std::is_signed<T2>::value));
    }
}
