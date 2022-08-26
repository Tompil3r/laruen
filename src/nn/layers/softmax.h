
#ifndef NN_LAYERS_SOFTMAX_H_
#define NN_LAYERS_SOFTMAX_H_

#include <algorithm>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Softmax : public Layer<T> {
            private:
                int_fast8_t axis_;
            
            public:
                static constexpr char NAME[] = "Softmax";

                Softmax(int_fast8_t axis = -1) noexcept : axis_(axis)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    uint_fast8_t real_axis = this->axis_ >= 0 ? this->axis_ : input.ndim() + this->axis_;
                    Shape sums_shape(input.shape());
                    sums_shape[real_axis] = 1;

                    NDArray<T> exp_sums(sums_shape);

                    input.exp(output);
                    output.sum({real_axis}, exp_sums);
                    output.divide_eq(exp_sums);

                    return output;
                }

                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {

                }

                void build(Shape::const_iterator begin, Shape::const_iterator end) override final {
                    this->output_shape_ = Shape(begin, end);
                }
                
                void build(const Shape &input_shape) override final {
                    this->output_shape_ = input_shape;
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }

                inline void axis(int_fast8_t axis) noexcept {
                    this->axis_ = axis;
                }

                inline int_fast8_t axis() const noexcept {
                    return this->axis_;
                }

                uint_fast64_t params() const noexcept override final {
                    return 0;
                }
        };
    }

    using namespace impl;
}


#endif