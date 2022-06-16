
#include <cstdint>
#include <string>
#include "src/ndlib/types.h"
#include "src/ndlib/range.h"

namespace laruen::ndlib {

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
