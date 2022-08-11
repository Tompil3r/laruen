
#ifndef NN_LAYERS_RELU_H_
#define NN_LAYERS_RELU_H_

#include "src/ndlib/ndarray.h"
#include "src/nn/layers/layer.h"


namespace laruen::nn::layers {
    namespace impl {

        using laruen::ndlib::NDArray;

        template <typename T>
        class ReLU : public Layer<T> {
            public:
                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.maximum(0, output);
                    return output;
                }

                void backward() const noexcept override final {
                }
        };
    }

    using namespace impl;
}

#endif