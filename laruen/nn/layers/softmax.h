
#ifndef LARUEN_NN_LAYERS_SOFTMAX_H_
#define LARUEN_NN_LAYERS_SOFTMAX_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/nditer.h"
#include "laruen/math/common.h"
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
        class Softmax : public Layer<T> {
            public:
                Softmax() noexcept
                {}

                inline Layer<T>* clone() const override final {
                    return new Softmax<T>(*this);
                }

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    NDIter input_iter(input.data(), input);
                    NDIter output_iter(output.data(), output);

                    for(uint_fast64_t i = 0;i < input.size() / input.shape().back();i++) {
                        T exp_sum = 0;
                        T max = input_iter.next();
                        uint_fast64_t j;

                        for(j = 1;j < input.shape().back();j++) {
                            max = laruen::math::common::max(max, input_iter.next());
                        }

                        input_iter.ptr -= input.dim_sizes().back();
                        input_iter.ndindex.back() = 0;

                        for(j = 0;j < input.shape().back();j++) {
                            output_iter.current() = std::exp(input_iter.next() - max);
                            exp_sum += output_iter.next();
                        }

                        output_iter.ptr -= output.dim_sizes().back();
                        output_iter.ndindex.back() = 0;

                        for(j = 0;j < output.shape().back();j++) {
                            output_iter.next() /= exp_sum;
                        }
                    }

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
                 * @param cached_input Z (unused)
                 * @param cached_output A (Softmax(Z))
                 * @param prev_grad_output dZ (dL / dZ)
                 */
                void backward(const NDArray<T> &grad, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_grad_output) noexcept override final
                {
                    const uint_fast64_t batch_size = grad.size() / grad.shape().back();
                    const uint_fast8_t grad_batch_axis = grad.ndim() - 2;
                    const uint_fast8_t cached_output_batch_axis = cached_output.ndim() - 2;

                    NDIter grad_iter(grad.data(), grad);
                    NDIter result_iter(prev_grad_output.data(), prev_grad_output);
                    NDIter cached_output_iter_i(cached_output.data(), cached_output);
                    NDIter cached_output_iter_j(cached_output.data(), cached_output);

                    for(uint_fast64_t k = 0;k < batch_size;k++) {

                        for(uint_fast64_t i = 0;i < grad.shape().back();i++) {
                            result_iter.current() = 0;

                            for(uint_fast64_t j = 0;j < grad.shape().back();j++) {
                                result_iter.current() += grad_iter.next() * cached_output_iter_i.current()
                                * ((i == j) - cached_output_iter_j.next());
                            }

                            cached_output_iter_i.next();
    
                            result_iter.next();
    
                            grad_iter.ptr -= grad.dim_sizes().back();
                            grad_iter.ndindex.back() = 0;

                            cached_output_iter_j.ptr -= cached_output.dim_sizes().back();
                            cached_output_iter_j.ndindex.back() = 0;
                        }

                        grad_iter.next(grad_batch_axis);
                        cached_output_iter_j.next(cached_output_batch_axis);
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
                    return "Softmax";
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

        template <typename T = float32_t>
        inline std::shared_ptr<Layer<T>> softmax() noexcept {
            return std::shared_ptr<Layer<T>>(new Softmax<T>());
        }
    }

    using namespace impl;
}


#endif