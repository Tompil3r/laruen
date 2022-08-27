
#ifndef NN_LAYERS_RELU_H_
#define NN_LAYERS_RELU_H_

#include <cassert>
#include <algorithm>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/multi/nditer.h"
#include "src/nn/layers/layer.h"


namespace laruen::nn::layers {
    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;
        using laruen::multi::NDIter;

        template <typename T = float32_t>
        class ReLU : public Layer<T> {
            public:
                static constexpr char NAME[] = "ReLU";

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.maximum(0, output);
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

                /**
                 * @brief calculates the gradient of the Loss function with respect to Z
                 * @param deriv dA (dL / dA) (dL = dLoss)
                 * @param cached_input Z
                 * @param cached_output A (ReLU(Z)) - not used in this case
                 * @param prev_deriv_output dZ (dL / dZ)
                 */
                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    assert(deriv.shape() == prev_deriv_output.shape());

                    NDIter deriv_iter(deriv.data(), deriv);
                    NDIter cached_input_iter(cached_input.data(), cached_input);
                    NDIter output_iter(prev_deriv_output.data(), prev_deriv_output);

                    /*
                        dL / dZ = (dL / dA) * (dA / dZ)
                        ReLU gradient : 0 if x <= 0 else 1.
                        Therefore the gradient multiplied by dA (dL / dA)
                        is 0 when x <= 0 and (dL / dA) otherwise:
                        (dA / dZ)[i] = Z[i] <= 0 : 0 (0 * (dL / dA)[i] = 0) ? (dL / dA) (1 * (dL / dA) = (dL / dA))

                        ** [i] denotes the i'th element
                    */
                    for(uint_fast64_t i = 0;i < deriv.size();i++) {
                        output_iter.next() = cached_input_iter.next() <= 0 ? 0 : deriv_iter.current();
                        deriv_iter.next();
                    }
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

                uint_fast64_t params() const noexcept override final {
                    return 0;
                }
        };
    }

    using namespace impl;
}

#endif