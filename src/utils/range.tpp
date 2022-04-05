
#include "src/ndarray/ndarray_types.h"
#include "src/utils/range.h"

#include <cstdint>
#include <ostream>

namespace laruen::utils {
    
    template <typename T>
    Range<T>::Range(T start, T end, T step) {
        this->start = start;
        this->end = end;
        this->step = step;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream &strm, const Range<T> &range) {
        strm << range.start << ':' << range.end << ':' << range.step;
        return strm;
    }
}
