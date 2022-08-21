
#ifndef NN_LAYERS_RELU_H_
#define NN_LAYERS_RELU_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/ndlib/nditer.h"
#include "src/nn/layers/layer.h"


namespace laruen::nn::layers {
    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;
        using laruen::ndlib::float32_t;
        using laruen::ndlib::NDIter;

        template <typename T = float32_t>
        class ReLU : public Layer<T> {
            public:
                static constexpr char NAME[] = "ReLU";

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.maximum(0, output);
                    return output;
                }

                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    assert(deriv.shape() == prev_deriv_output.shape());

                    NDIter deriv_iter(deriv.data(), deriv);
                    NDIter pdo_iter(prev_deriv_output.data(), prev_deriv_output);

                    for(uint_fast64_t i = 0;i < deriv.size();i++) {
                        pdo_iter.next() = deriv_iter.next() > 0 ? 1 : 0;
                    }
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