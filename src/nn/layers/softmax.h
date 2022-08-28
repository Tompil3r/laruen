
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
            public:
                static constexpr char NAME[] = "Softmax";

                Softmax() noexcept
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    assert(input.ndim() == 2);

                    NDArray<T> exp_sums(Shape{input.shape().front(), 1});

                    input.exp(output);
                    output.sum({1}, exp_sums);
                    output.divide_eq(exp_sums);

                    return output;
                }

                NDArray<T> forward(const NDArray<T> &input) override final {
                    if(!std::equal(input.shape().cbegin() + 1, input.shape().cend(), this->output_shape_.cbegin())) {
                        this->build(input.shape().cbegin() + 1, input.shape().cend());
                    }

                    NDArray<T> output(input.shape());
                    this->forward(input, output);

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

                inline void compile(uint_fast64_t required_caches) override final
                {}

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