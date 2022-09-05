
#ifndef MULTI_RANGE_H_
#define MULTI_RANGE_H_

#include <string>
#include <ostream>

namespace laruen::multi {
    
    template <typename T> struct Range {
        T start;
        T end;
        T step;

        inline constexpr Range(T end) noexcept
        : start(0), end(end), step(1)
        {}

        inline constexpr Range(T start, T end) noexcept
        : start(start), end(end), step(1)
        {}

        inline constexpr Range(T start, T end, T step) noexcept
        : start(start), end(end), step(step)
        {}
        
        std::string str() const noexcept {
            return std::to_string(this->start) + ':' +
            std::to_string(this->end) + ':' + std::to_string(this->step);
        }

        friend inline std::ostream& operator<<(std::ostream &stream, const Range<T> &range) noexcept {
            return stream << range.str();
        }
    };
}

#endif