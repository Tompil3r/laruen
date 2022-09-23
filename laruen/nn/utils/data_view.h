
#ifndef LARUEN_NN_UTILS_DATA_VIEW_H_
#define LARUEN_NN_UTILS_DATA_VIEW_H_

#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/utils/utils.h"

namespace laruen::nn::utils {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::utils::batch_view;

        template <typename T = float32_t>
        struct DataView {
            public:
                uint_fast64_t full_batches = 0;
                uint_fast64_t remaining = 0;
                uint_fast64_t batches = 0;
                uint_fast64_t x_batch_stride;
                uint_fast64_t y_batch_stride;
                T remaining_ratio;
                T partial_batches;
                const NDArray<T> x_batch;
                const NDArray<T> y_batch;
                const NDArray<T> x_remaining;
                const NDArray<T> y_remaining;

                DataView(const NDArray<T> &x, const NDArray<T> &y, uint_fast64_t batch_size) {
                    if(!x.size()) {
                        return;
                    }

                    this->full_batches = x.shape().front() / batch_size;
                    this->remaining = x.shape().front() % batch_size;
                    this->batches = this->full_batches + (this->remaining > 0);
                    this->x_batch_stride = batch_size * x.strides().front();
                    this->y_batch_stride = batch_size * y.strides().front();
                    this->remaining_ratio = this->full_batches > 0 ? (T)this->remaining / batch_size : 1;
                    this->partial_batches = this->full_batches + this->remaining_ratio;

                    if(this->full_batches) {
                        x_batch = batch_view(x, batch_size);
                        y_batch = batch_view(y, batch_size);
                    }

                    if(this->remaining) {
                        this->x_remaining = batch_view(x, this->remaining);
                        this->y_remaining = batch_view(y, this->remaining);
                        this->x_remaining.data(x.data() + this->full_batches * x_batch_stride);
                        this->y_remaining.data(y.data() + this->full_batches * y_batch_stride);
                    }
                }

                inline void next_x_batch() noexcept {
                    this->x_batch.data(this->x_batch.data() + x_batch_stride);
                }

                inline void next_y_batch() noexcept {
                    this->y_batch.data(this->y_batch.data() + y_batch_stride);
                }

                inline void next_batch() noexcept {
                    this->next_x_batch();
                    this->next_y_batch();
                }

                inline void reset_x_batch(const T *x_data) noexcept {
                    this->x_batch.data(x_data);
                }

                inline void reset_y_batch(const T *y_data) noexcept {
                    this->y_batch.data(y_data);
                }

                inline void reset_batch_views(const T *x_data, const T *y_data) noexcept {
                    this->reset_x_batch(x_data);
                    this->reset_y_batch(y_data);
                }
        };
    }

    using namespace impl;
}

#endif