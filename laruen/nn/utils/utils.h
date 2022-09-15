
#ifndef LARUEN_NN_UTILS_UTILS_H_
#define LARUEN_NN_UTILS_UTILS_H_

#include <cstdint>
#include <algorithm>
#include <utility>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"

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
        
        template <typename T>
        NDArray<T> batch_view(NDArray<T> &ndarray, uint_fast64_t batch_size) {
            NDArray<T> view = ndarray.view();

            view.size() = (ndarray.size() / ndarray.shape().front()) * batch_size;
            view.shape().front() = batch_size;
            view.dim_sizes().front() = (ndarray.dim_sizes().front() / ndarray.shape().front()) * batch_size;

            return view;
        }

        template <typename T>
        inline constexpr T stable_nonzero(T num, T threshold = 1e-12) noexcept {
            return std::abs(num) >= threshold ? num : (num >= 0 ? threshold : -threshold);
        }
    }

    using namespace impl;
}

#endif