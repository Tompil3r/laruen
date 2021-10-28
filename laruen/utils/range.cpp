
#include "laruen/ndarray/typenames.h"
#include "laruen/utils/range.h"

#include <cstdint>

using namespace laruen::utils;

template class Range<int8_t>;
template class Range<uint8_t>;
template class Range<int16_t>;
template class Range<uint16_t>;
template class Range<int32_t>;
template class Range<uint32_t>;
template class Range<int64_t>;
template class Range<uint64_t>;
template class Range<float32_t>;
template class Range<float64_t>;


template <typename T> Range<T>::Range(T start, T end, T step)
{
    this->start = start;
    this->end = end;
    this->step = step;
}
