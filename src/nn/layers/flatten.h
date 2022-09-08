
#ifndef LARUEN_NN_LAYERS_FLATTEN_H_
#define LARUEN_NN_LAYERS_FLATTEN_H_

#include <numeric>
#include <functional>
#include <cstdint>
#include <memory>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/layers/layer.h"
#include "src/nn/optimizers/optimizer.h"

namespace laruen::nn::layers {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;
        using laruen::nn::optimizers::Optimizer;

        template <typename T = float32_t>
        class Flatten : public Layer<T> {
            uint_fast64_t size_;

            public:
                static constexpr char NAME[] = "Flatten";

                Flatten() noexcept : size_(0)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (dim0, dim1, ...)
                    // output.shape = (dim0, dim1 * ...)
                    
                    output = input.copy_reshape({input.shape().front(), this->size_});
                    return output;
                }

                NDArray<T> forward(const NDArray<T> &input) override final {
                    if(input.size() / input.shape().front() != this->size_) {
                        this->build(input.shape().cbegin() + 1, input.shape().cend());
                    }

                    NDArray<T> output;
                    this->forward(input, output);

                    return output;
                }

                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    assert(deriv.size() == prev_deriv_output.size());
                    prev_deriv_output.copy_data_from(deriv);
                }

                inline void update_weights(const Optimizer<T> &optimizer) override final
                {}

                void build(Shape::const_iterator begin, Shape::const_iterator end) override final {
                    // input_shape = (dim1, ...)
                    this->size_ = std::accumulate(begin, end, 1, std::multiplies<T>{});
                    this->output_shape_ = {this->size_};
                }

                inline void build(const Shape &input_shape) override final {
                    this->build(input_shape.begin(), input_shape.end());
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }

                uint_fast64_t params() const noexcept override final {
                    return 0;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Layer<T>> shared_flatten() noexcept {
            return std::shared_ptr<Layer<T>>(new Flatten<T>());
        }
    }

    using namespace impl;
}

#endif