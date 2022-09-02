
#ifndef NN_UTILS_H_
#define NN_UTILS_H_

#include <cstdint>
#include <algorithm>
#include <utility>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"

namespace laruen::nn::utils {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::Strides;

        Shape add_batch_shape(const Shape &shape, uint_fast64_t batch_size) noexcept {
            Shape result(shape.size() + 1);
            std::copy(shape.cbegin(), shape.cend(), result.begin() + 1);
            result.front() = batch_size;

            return result;
        }

        Shape replace_batch_shape(const Shape &shape, uint_fast64_t batch_size) noexcept {
            Shape result(shape);
            result.front() = batch_size;
            return result;
        }

        template <typename T>
        const NDArray<T> batch_view(const NDArray<T> &ndarray, uint_fast64_t batch_size) {
            const NDArray<T> view = ndarray.view();

            view.size() = (ndarray.size() / ndarray.shape().front()) * batch_size;
            view.shape().front() = batch_size;
            view.dim_sizes().front() = (ndarray.dim_sizes().front() / ndarray.shape().front()) * batch_size;

            return view;
        }
    }

    using namespace impl;
}

#endif