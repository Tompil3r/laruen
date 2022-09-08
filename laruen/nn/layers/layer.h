
#ifndef LARUEN_NN_LAYERS_LAYER_H_
#define LARUEN_NN_LAYERS_LAYER_H_

#include <cassert>
#include <utility>
#include <fstream>
#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/optimizers/optimizer.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::multi::Shape;
        using laruen::nn::optimizers::Optimizer;

        template <typename T = float32_t>
        class Layer {
            protected:
                Shape output_shape_;

            public:
                virtual ~Layer()
                {}

                Layer() noexcept = default;

                Layer(const Shape &output_shape) noexcept : output_shape_(output_shape)
                {}
                
                const Shape& output_shape() const noexcept {
                    return this->output_shape_;
                }

                inline NDArray<T> operator()(const NDArray<T> &input) {
                    return this->forward(input);
                }

                inline virtual void save_weights(std::ofstream &file, int_fast64_t offset = 0) const
                {}

                inline virtual void load_weights(std::ifstream &file, int_fast64_t offset = 0)
                {}

                inline virtual void compile(uint_fast64_t required_caches)
                {};

                virtual NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &out) const = 0;

                virtual NDArray<T> forward(const NDArray<T> &input) = 0;

                virtual void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept = 0;

                virtual void update_weights(const Optimizer<T> &optimizer) = 0;

                virtual void build(const Shape &input_shape) = 0;

                virtual void build(Shape::const_iterator begin, Shape::const_iterator end) = 0;

                virtual const char* name() const noexcept = 0;

                virtual uint_fast64_t params() const noexcept = 0;
        };
    }

    using namespace impl;
}


#endif