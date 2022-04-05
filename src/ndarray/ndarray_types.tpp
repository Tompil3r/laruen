
#include "src/ndarray/ndarray_types.h"
#include <string>
#include <tuple>
#include <cstdint>

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
    template <typename T, typename... Types>
    struct Type<T, std::tuple<T, Types...>> {
        static constexpr uint64_t id = 1;
    };

    template <typename T, typename U, typename... Types>
    struct Type<T, std::tuple<U, Types...>> {
        static constexpr uint64_t id = 1 + Type<T, std::tuple<Types...>>::id;
    };

    template <typename T>
    constexpr uint64_t type_id() {
        return Type<T, std::tuple<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
        int64_t, uint64_t, float32_t, float64_t>>::id;
    }

    constexpr bool type_signed(uint64_t type_id) {
        return type_id % 2;
    }

    constexpr bool type_decimal(uint64_t type_id) {
        return type_id > 8;
    }

    template <typename T, typename U>
    constexpr bool type_contained() {
        constexpr uint64_t id_t = type_id<T>();
        constexpr uint64_t id_u = type_id<U>();

        return id_t >= id_u && (type_decimal(id_t) || (type_signed(id_t) || type_signed(id_t) == type_signed(id_u)));
    }
}
