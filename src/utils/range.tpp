
#include "src/ndarray/ndarray_types.h"
#include "src/utils/range.h"

#include <cstdint>
#include <string>

namespace laruen::utils {

    template <typename T>
    constexpr Range<T>::Range(T start, T end, T step) {
        this->start = start;
        this->end = end;
        this->step = step;
    }

    template <typename T>
    std::string Range<T>::str() const {
        return std::to_string(this->start) + ':' + std::to_string(this->end) + ':' + std::to_string(this->step);
    }
}
