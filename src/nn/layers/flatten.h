
#ifndef NN_LAYERS_FLATTEN_H_
#define NN_LAYERS_FLATTEN_H_

#include <numeric>
#include <functional>
#include <cstdint>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class Flatten : public Layer<T> {
            uint_fast64_t size_;

            public:
                static constexpr char NAME[] = "Flatten";

                Flatten() noexcept : size_(0)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (dim0, dim1, ...)
                    // output.shape = (dim0, dim1 * ...)
                    
                    assert(input.contig());
                    output = input.view_reshape({input.shape()[0], this->size_});
                    return output;
                }

                void backward() const noexcept override final
                {}

                Shape build(const Shape &input_shape) override final {
                    // input_shape = (number samples = 0, dim1, ...)
                    this->size_ = std::accumulate(input_shape.cbegin() + 1, input_shape.cend(), 1, std::multiplies<T>{});

                    return Shape{this->size_};
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }
        };

    }

    using namespace impl;
}

#endif