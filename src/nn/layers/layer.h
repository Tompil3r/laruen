
#ifndef NN_LAYERS_LAYER_H_
#define NN_LAYERS_LAYER_H_

#include <cassert>
#include <utility>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class Layer {

            public:
                void forward() const noexcept = 0;
                void backward() const noexcept = 0;
        };
    }

    using namespace impl;
}


#endif