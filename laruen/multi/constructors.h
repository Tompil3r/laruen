
#ifndef LARUEN_MULTI_CONSTRUCTORS_H_
#define LARUEN_MULTI_CONSTRUCTORS_H_

#include <initializer_list>
#include <numeric>
#include <functional>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/range.h"

namespace laruen::multi {
    template <typename> class NDArray;

    template <typename T = float32_t>
    inline NDArray<T> empty(const Shape &shape) {
        return NDArray<T>(shape);
    }

    template <typename T = float32_t>
    inline NDArray<T> zeros(const Shape &shape) {
        return NDArray<T>(shape, (T)0);
    }

    template <typename T = float32_t>
    inline NDArray<T> ones(const Shape &shape) {
        return NDArray<T>(shape, (T)1);
    }

    template <typename T>
    inline NDArray<T> full(const Shape &shape, T value) {
        return NDArray<T>(shape, value);
    }

    template <typename T = float32_t>
    inline NDArray<T> random_uniform(const Shape &shape) {
        NDArray<T> array(shape);
        array.random_uniform((T)0, (T)1);
        return array;
    }

    template <typename T = float32_t>
    inline NDArray<T> random_uniform(const Shape &shape, T min, T max) {
        NDArray<T> array(shape);
        array.random_uniform(min, max);
        return array;
    }

    template <typename T>
    inline NDArray<T> random_uniform(const Shape &shape, T max) {
        NDArray<T> array(shape);
        array.random_uniform((T)0, max);
        return array;
    }
    
    template <typename T>
    inline NDArray<T> randint(const Shape &shape, T min, T max) {
        NDArray<T> array(shape);
        array.randint(min, max);
        return array;
    }

    template <typename T>
    inline NDArray<T> randint(const Shape &shape, T max) {
        NDArray<T> array(shape);
        array.randint((T)0, max);
        return array;
    }

    template <typename T>
    inline NDArray<T> range(const Shape &shape, const Range<T> &range) {
        return NDArray<T>(shape, range);
    }

    template <typename T>
    inline NDArray<T> range(const Range<T> &range) {
        return NDArray<T>(range);
    }

    template <typename T = float32_t>
    inline NDArray<T> range(const Shape &shape) {
        return NDArray<T>(shape,
        Range<T>(std::accumulate(shape.cbegin(), shape.cend(), (T)1, std::multiplies<T>{})));
    }

    template <typename T>
    inline NDArray<T> array(std::initializer_list<T> init_list, const Shape &shape) {
        return NDArray<T>(init_list, shape);
    }

    template <typename T = float32_t>
    inline NDArray<T> array(std::initializer_list<T> init_list = {}) {
        return NDArray<T>(init_list);
    }
}

#endif