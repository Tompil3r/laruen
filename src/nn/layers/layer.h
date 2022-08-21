
#ifndef NN_LAYERS_LAYER_H_
#define NN_LAYERS_LAYER_H_

#include <cassert>
#include <utility>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::ndlib::NDArray;
        using laruen::ndlib::float32_t;
        using laruen::ndlib::Shape;

        template <typename T = float32_t>
        class Layer {
            protected:
                Shape output_shape_;

            public:
                Layer() noexcept = default;

                Layer(const Shape &output_shape) noexcept : output_shape_(output_shape)
                {}
                
                const Shape& output_shape() const noexcept {
                    return this->output_shape_;
                }

                virtual NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &out) const = 0;
                virtual void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept = 0;
                virtual void build(const Shape &input_shape) = 0;
                virtual const char* name() const noexcept = 0;
                virtual uint_fast64_t params() const noexcept = 0;
        };
    }

    using namespace impl;
}


#endif