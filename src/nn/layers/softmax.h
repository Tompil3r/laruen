
#ifndef NN_LAYERS_SOFTMAX_H_
#define NN_LAYERS_SOFTMAX_H_

#include <algorithm>
#include <cmath>
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
        class Softmax : public Layer<T> {
            public:
                static constexpr char NAME[] = "Softmax";

                Softmax() noexcept
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    NDIter input_iter(input.data(), input);
                    NDIter output_iter(output.data(), output);

                    for(uint_fast64_t i = 0;i < input.size() / input.shape().back();i++) {
                        T exp_sum = 0;
                        uint_fast64_t j;

                        for(j = 0;j < input.shape().back();j++) {
                            output_iter.current() = std::exp(input_iter.next());
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

                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    const uint_fast64_t batch_size = deriv.size() / deriv.shape().back();
                    const uint_fast8_t deriv_batch_axis = deriv.ndim() - 2;
                    const uint_fast8_t cached_output_batch_axis = cached_output.ndim() - 2;

                    NDIter deriv_iter(deriv.data(), deriv);
                    NDIter result_iter(prev_deriv_output.data(), prev_deriv_output);
                    NDIter cached_output_iter_i(cached_output.data(), cached_output);
                    NDIter cached_output_iter_j(cached_output.data(), cached_output);

                    /**
                     * @brief calculates the gradient of the Loss function with respect to Z
                     * @param deriv dA (dL / dA) (dL = dLoss)
                     * @param cached_input Z (unused)
                     * @param cached_output A (Softmax(Z))
                     * @param prev_deriv_output dZ (dL / dZ)
                     */
                    for(uint_fast64_t k = 0;k < batch_size;k++) {

                        for(uint_fast64_t i = 0;i < deriv.shape().back();i++) {
                            result_iter.current() = 0;

                            for(uint_fast64_t j = 0;j < deriv.shape().back();j++) {
                                result_iter.current() += deriv_iter.next() * cached_output_iter_i.current()
                                * ((i == j) - cached_output_iter_j.next());
                            }

                            cached_output_iter_i.next();
    
                            result_iter.next();
    
                            deriv_iter.ptr -= deriv.dim_sizes().back();
                            deriv_iter.ndindex.back() = 0;

                            cached_output_iter_j.ptr -= cached_output.dim_sizes().back();
                            cached_output_iter_j.ndindex.back() = 0;
                        }

                        deriv_iter.next(deriv_batch_axis);
                        cached_output_iter_j.next(cached_output_batch_axis);
                    }
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