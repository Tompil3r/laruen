
#ifndef MULTI_SLICING_RANGE_H_
#define MULTI_SLICING_RANGE_H_

#include <cstdint>
#include <string>
#include <ostream>

namespace laruen::multi {
    
    struct SlicingRange {
        uint_fast64_t start;
        uint_fast64_t end;
        uint_fast64_t step;

        inline constexpr SlicingRange(uint_fast64_t index) noexcept
        : start(index), end(index + 1), step(1)
        {}

        inline constexpr SlicingRange(uint_fast64_t start, uint_fast64_t end) noexcept
        : start(start), end(end), step(1)
        {}

        inline constexpr SlicingRange(uint_fast64_t start, uint_fast64_t end, uint_fast64_t step) noexcept
        : start(start), end(end), step(step)
        {}
        
        std::string str() const noexcept {
            std::string str(std::to_string(this->start));
            str.push_back(':');
            str.append(std::to_string(this->end));
            str.push_back(':');
            str.append(std::to_string(this->step));

            return str;
        }

        friend inline std::ostream& operator<<(std::ostream &stream, const SlicingRange &range) noexcept {
            return stream << range.str();
        }
    };
}

#endif