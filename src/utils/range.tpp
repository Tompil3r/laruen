
#include "src/ndlib/ndlib_types.h"
#include "src/utils/range.h"

#include <cstdint>
#include <string>

namespace laruen::utils {

    template <typename T>
    constexpr Range<T>::Range(T end) noexcept {
        this->start = 0;
        this->end = end;
        this->step = 1;
    }

    template <typename T>
    constexpr Range<T>::Range(T start, T end) noexcept {
        this->start = start;
        this->end = end;
        this->step = 1;
    }

    template <typename T>
    constexpr Range<T>::Range(T start, T end, T step) noexcept {
        this->start = start;
        this->end = end;
        this->step = step;
    }

    template <typename T>
    std::string Range<T>::str() const noexcept {
        return std::to_string(this->start) + ':' + std::to_string(this->end) + ':' + std::to_string(this->step);
    }
}
