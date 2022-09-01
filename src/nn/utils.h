
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
        NDArray<T> batch_view(NDArray<T> &ndarray, uint_fast64_t batch_size) {
            Shape new_shape(ndarray.shape());
            Strides new_dim_sizes(ndarray.dim_sizes());

            uint_fast64_t new_size = (ndarray.size() / ndarray.shape().front()) * batch_size;
            new_shape.front() = batch_size;
            new_dim_sizes.front() = (new_dim_sizes.front() / ndarray.shape().front()) * batch_size;

            return NDArray<T>(ndarray.data(), std::move(new_shape), Strides(ndarray.strides()),
            std::move(new_dim_sizes), new_size, ndarray.ndim(), ndarray.contig(), false);
        }
    }

    using namespace impl;
}

#endif