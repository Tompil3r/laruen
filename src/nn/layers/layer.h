
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

            public:
                virtual NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &out) const = 0;
                virtual void backward() const noexcept = 0;
                virtual Shape build(const Shape &input_shape) = 0;
                virtual const char* name() const noexcept = 0;
        };
    }

    using namespace impl;
}


#endif