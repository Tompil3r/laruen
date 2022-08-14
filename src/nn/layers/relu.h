
#ifndef NN_LAYERS_RELU_H_
#define NN_LAYERS_RELU_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"


namespace laruen::nn::layers {
    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;

        template <typename T>
        class ReLU : public Layer<T> {
            public:
                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.maximum(0, output);
                    return output;
                }

                void backward() const noexcept override final {
                }

                Shape build(const Shape &input_shape) override final {
                    return input_shape;
                }
        };
    }

    using namespace impl;
}

#endif