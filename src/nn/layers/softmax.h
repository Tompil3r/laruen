
#ifndef NN_LAYERS_SOFTMAX_H_
#define NN_LAYERS_SOFTMAX_H_

#include <algorithm>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;

        template <typename T>
        class Softmax : public Layer<T> {
            private:
                int_fast8_t axis_;
            
            public:
                Softmax(int_fast8_t axis = -1) noexcept : axis_(axis)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &out) const override final {
                    uint_fast8_t real_axis = this->axis_ >= 0 ? this->axis_ : input.ndim() + this->axis_;
                    Shape sums_shape(input.shape());
                    sums_shape[real_axis] = 1;

                    NDArray<T> exp_sums(sums_shape);

                    input.exp(out);
                    out.sum({real_axis}, exp_sums);
                    out.divide_eq(exp_sums);

                    return out;
                }

                void backward() const noexcept override final {

                }

                inline void axis(int_fast8_t axis) noexcept {
                    this->axis_ = axis;
                }

                inline int_fast8_t axis() const noexcept {
                    return this->axis_;
                }
        };
    }

    using namespace impl;
}


#endif