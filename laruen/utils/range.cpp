
#include "laruen/ndarray/ndarray_typenames.h"
#include "laruen/utils/range.h"

#include <cstdint>

using namespace laruen::utils;

template struct Range<int8_t>;
template struct Range<uint8_t>;
template struct Range<int16_t>;
template struct Range<uint16_t>;
template struct Range<int32_t>;
template struct Range<uint32_t>;
template struct Range<int64_t>;
template struct Range<uint64_t>;
template struct Range<float32_t>;
template struct Range<float64_t>;


template <typename T> Range<T>::Range(T start, T end, T step)
{
    this->start = start;
    this->end = end;
    this->step = step;
}
