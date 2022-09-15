
#ifndef LARUEN_NN_UTILS_DATA_INFO_H_
#define LARUEN_NN_UTILS_DATA_INFO_H_

#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"

namespace laruen::nn::utils {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        struct DataInfo {
            public:
                uint_fast64_t full_batches;
                uint_fast64_t remaining;
                uint_fast64_t batches;
                uint_fast64_t x_batch_stride;
                uint_fast64_t y_batch_stride;
                T remaining_ratio;
                T partial_batches;

                DataInfo(uint_fast64_t samples, uint_fast64_t x_samples_stride,
                uint_fast64_t y_samples_stride, uint_fast64_t batch_size)
                :
                full_batches(samples / batch_size),
                remaining(samples % batch_size),
                batches(this->full_batches + (this->remaining > 0)),
                x_batch_stride(batch_size * x_samples_stride),
                y_batch_stride(batch_size * y_samples_stride),
                remaining_ratio(this->full_batches > 0 ? (T)this->remaining / batch_size : 1),
                partial_batches(this->full_batches + this->remaining_ratio)
                {}

                DataInfo(const NDArray<T> &x, const NDArray<T> &y, uint_fast64_t batch_size)
                : DataInfo(x.shape().front(), x.strides().front(), y.strides().front(), batch_size)
                {}
        };
    }

    using namespace impl;
}

#endif