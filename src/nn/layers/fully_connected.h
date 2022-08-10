
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
                    w.shape = (nodes, inputs)
                    b.shape = (inputs)
                    dw.shape = (nodes, inputs)
                    db.shape = (inputs)
                */
                NDArray<T> w;
                NDArray<T> b;
                NDArray<T> dw;
                NDArray<T> db;

            public:
                FullyConnected(const Shape &shape) noexcept
                : w(shape, 0, 1), b({shape[1]}, 0), dw(w.shape()), db(b.shape())
                {
                    // shape should be (nodes, inputs)
                    assert(shape.size() == 2);
                    // w - initialized with random numbers in range 0-1
                    // b - initialized with 0s
                    // dw - uninitialized
                    // db - uninitialized
                    // (all shapes initialized, not all data)
                }

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (number samples, inputs)
                    // out.shape = (number samples, nodes)
                    input.matmul(this->w, output);
                    out.add(b);
                    
                    return output;
                }

                void backward() const noexcept override final {
                    
                }
        };
    }

    using namespace impl;
}




#endif