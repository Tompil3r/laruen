
#ifndef LARUEN_NN_LAYERS_RELU_H_
#define LARUEN_NN_LAYERS_RELU_H_

#include <cassert>
#include <algorithm>
#include <memory>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/nditer.h"
#include "laruen/nn/layers/layer.h"
#include "laruen/nn/optimizers/optimizer.h"


namespace laruen::nn::layers {
    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;
        using laruen::multi::NDIter;
        using laruen::nn::optimizers::Optimizer;

        template <typename T = float32_t>
        class ReLU : public Layer<T> {
            public:
                inline Layer<T>* clone() const override final {
                    return new ReLU<T>(*this);
                }

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.maximum(0, output);
                    return output;
                }

                NDArray<T> forward(const NDArray<T> &input) override final {
                    if(input.ndim() != this->output_shape_.size() ||
                        !std::equal(input.shape().cbegin() + 1, input.shape().cend(), this->output_shape_.cbegin()))
                    {
                        this->build(input.shape().cbegin() + 1, input.shape().cend());
                    }

                    NDArray<T> output(input.shape());
                    this->forward(input, output);

                    return output;
                }

                /**
                 * @brief calculates the gradient of the Loss function with respect to Z
                 * @param grad dA (dL / dA) (dL = dLoss)
                 * @param cached_input Z
                 * @param cached_output A (ReLU(Z)) - not used in this case
                 * @param prev_grad_output dZ (dL / dZ)
                 */
                void backward(const NDArray<T> &grad, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_grad_output) noexcept override final
                {
                    assert(grad.shape() == prev_grad_output.shape());

                    NDIter grad_iter(grad.data(), grad);
                    NDIter cached_input_iter(cached_input.data(), cached_input);
                    NDIter output_iter(prev_grad_output.data(), prev_grad_output);

                    /*
                        dL / dZ = (dL / dA) * (dA / dZ)
                        ReLU gradient : 0 if x <= 0 else 1.
                        Therefore the gradient multiplied by dA (dL / dA)
                        is 0 when x <= 0 and (dL / dA) otherwise:
                        (dA / dZ)[i] = Z[i] <= 0 : 0 (0 * (dL / dA)[i] = 0) ? (dL / dA) (1 * (dL / dA) = (dL / dA))

                        ** [i] denotes the i'th element
                    */
                    for(uint_fast64_t i = 0;i < grad.size();i++) {
                        output_iter.next() = cached_input_iter.next() <= 0 ? 0 : grad_iter.current();
                        grad_iter.next();
                    }
                }

                inline void update_weights(const Optimizer<T> &optimizer) override final
                {}

                void build(Shape::const_iterator begin, Shape::const_iterator end) override final {
                    this->output_shape_ = Shape(begin, end);
                }

                void build(const Shape &input_shape) override final {
                    this->output_shape_ = input_shape;
                }

                const char* name() const noexcept override final {
                    return "ReLU";
                }

                uint_fast64_t params() const noexcept override final {
                    return 0;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Layer<T>> relu() noexcept {
            return std::shared_ptr<Layer<T>>(new ReLU<T>());
        }
    }

    using namespace impl;
}

#endif