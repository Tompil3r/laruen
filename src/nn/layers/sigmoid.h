
#ifndef NN_LAYERS_SIGMOID_H_
#define NN_LAYERS_SIGMOID_H_

#include "src/ndlib/ndarray.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::ndlib::NDArray;

        template <typename T>
        class Sigmoid : public Layer<T> {
            
            public:
                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &out) const override final {
                    input.negate(out);
                    out.exp_eq();
                    out.add_eq(1);
                    out.inverse_divide_eq(1);

                    return out;
                }

                void backward() const noexcept override final {

                }
        };
    }

    using namespace impl;
}

#endif