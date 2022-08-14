
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
                static constexpr char NAME[] = "Fully Connected";

                FullyConnected(uint_fast32_t nodes) noexcept
                : nodes_(nodes)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (number samples, inputs)
                    // output.shape = (number samples, nodes)
                    input.matmul(this->w_, output);
                    output.add(this->b_);
                    
                    return output;
                }

                void backward() const noexcept override final {
                    
                }

                Shape build(const Shape &input_shape) override final {
                    // input_shape = (number samples = 0, inputs)
                    assert(input_shape.size() == 2);

                    this->w_ = NDArray<T>({input_shape[1], this->nodes_}, -1, 1);
                    this->b_ = NDArray<T>({this->nodes_}, 0);
                    this->dw_ = NDArray<T>(this->w_.shape());
                    this->db_ = NDArray<T>(this->b_.shape());

                    return Shape{0, this->nodes_}; // 0 - number of samples - unknown
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }
        };
    }

    using namespace impl;
}




#endif