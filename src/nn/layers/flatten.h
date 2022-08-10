
#ifndef NN_LAYERS_FLATTEN_H_
#define NN_LAYERS_FLATTEN_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {


        template <typename T>
        class Flatten : public Layer<T> {
            public:

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (samples, dim1, ...)
                    // output.shape = (samples, dim1 * ...)
                    
                    // ***implement***
                    return out;
                }

                void backward() const noexcept override final {

                }
        }

    }

    using namespace impl;
}

#endif