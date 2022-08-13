
#ifndef NN_LAYERS_FULLY_CONNECTED_H_
#define NN_LAYERS_FULLY_CONNECTED_H_

#include <cassert>
#include <utility>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class FullyConnected : public Layer<T> {
            private:
                /*
                    w.shape = (inputs, nodes)
                    b.shape = (nodes)
                    dw.shape = (inputs, nodes)
                    db.shape = (nodes)
                */
                NDArray<T> w_;
                NDArray<T> b_;
                NDArray<T> dw_;
                NDArray<T> db_;
                uint_fast32_t nodes_;

            public:
                FullyConnected(uint_fast32_t nodes) noexcept
                : nodes_(nodes)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (number samples, inputs)
                    // out.shape = (number samples, nodes)
                    input.matmul(this->w_, output);
                    output.add(this->b_);
                    
                    return output;
                }

                void backward() const noexcept override final {
                    
                }
        };
    }

    using namespace impl;
}




#endif