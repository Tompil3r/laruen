
#ifndef NN_LAYERS_SIGMOID_H_
#define NN_LAYERS_SIGMOID_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::ndlib::NDArray;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class Sigmoid : public Layer<T> {
            
            public:
                static constexpr char NAME[] = "Sigmoid";

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.negate(output);
                    output.exp_eq();
                    output.add_eq(1);
                    output.inverse_divide_eq(1);

                    return output;
                }

                void backward() const noexcept override final {

                }

                Shape build(const Shape &input_shape) override final {
                    return input_shape;
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }
        };
    }

    using namespace impl;
}

#endif