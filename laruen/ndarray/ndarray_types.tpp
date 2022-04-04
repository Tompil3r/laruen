
#include "laruen/ndarray/ndarray_types.h"
#include <ostream>

// operator<< for Shape is used as operator<< for NDIndex and Strides 
std::ostream& operator<<(std::ostream &strm, const Shape &shape) {
    uint8_t size = shape.size();
    strm << '(';

    for(uint8_t i = 0;i < size - 1;i++) {
        strm << shape[i];
        strm << ", ";
    }

    strm << shape[size - 1];
    strm << ')';

    return strm;
}

std::ostream& operator<<(std::ostream &strm, const SliceRanges &slice_ranges) {
    uint8_t size = slice_ranges.size();
    strm << '(';

    for(uint8_t i = 0;i < size - 1;i++) {
        strm << slice_ranges[i];
        strm << ", ";
    }

    strm << slice_ranges[size - 1];
    strm << ')';

    return strm;
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
}
