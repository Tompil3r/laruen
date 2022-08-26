
#ifndef NN_LAYERS_SIGMOID_H_
#define NN_LAYERS_SIGMOID_H_

#include <cassert>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/multi/nditer.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::multi::NDIter;

        template <typename T = float32_t>
        class Sigmoid : public Layer<T> {
            
            public:
                static constexpr char NAME[] = "Sigmoid";

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    input.negate(output);
                    output.exp_eq();
                    output.add_eq(1);
                    output.inverse_divide_eq(1);

                    return output;
                }

                /**
                 * @brief calculates the gradient of the Loss function with respect to Z
                 * @param deriv dA (dL / dA) (dL = dLoss)
                 * @param cached_input Z - not used in this case since (dL / dZ) can be
                 * calculated through A (Sigmoid(Z)) more efficiently -> dS(z)/dZ = S(Z) * (1 - S(Z))
                 * @param cached_output A (Sigmoid(Z))
                 * @param prev_deriv_output dZ (dL / dZ)
                 */
                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    assert(deriv.shape() == prev_deriv_output.shape());

                    NDIter deriv_iter(deriv.data(), deriv);
                    NDIter cached_output_iter(cached_output.data(), cached_output);
                    NDIter output_iter(prev_deriv_output.data(), prev_deriv_output);

                    /*
                        (dL / dZ)[i] = (dL / dA)[i] * A[i] * (1 - A[i])
                        ** [i] denotes the i'th element
                    */
                    for(uint_fast64_t i = 0;i < deriv.size();i++) {
                        output_iter.next() = deriv_iter.next() * cached_output_iter.current()
                        * (1 - cached_output_iter.current());

                        cached_output_iter.next();
                    }
                }

                void build(const Shape &input_shape) override final {
                    this->output_shape_ = input_shape;
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }

                uint_fast64_t params() const noexcept override final {
                    return 0;
                }
        };
    }

    using namespace impl;
}

#endif