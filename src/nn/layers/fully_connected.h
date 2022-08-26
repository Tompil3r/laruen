
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
                NDArray<T> raw_dw_;
                NDArray<T> raw_db_;
                NDArray<T> final_dw_;
                NDArray<T> final_db_;
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

                /**
                 * @brief calculates the gradient of the Loss function with respect to A
                 * @param deriv dZ (dL / dZ) (dL = dLoss)
                 * @param cached_input A
                 * @param cached_output Z (A * W) (matmul)
                 * @param prev_deriv_output dA (dL / dA)
                 */
                void backward(const NDArray<T> &deriv, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_deriv_output) noexcept override final
                {
                    /*
                        deriv.shape = (batch_size, nodes)
                        cached_input.shape = (batch_size, nb_inputs)
                        cached_output.shape = (batch_size, nodes)
                        prev_deriv_output.shape = (batch_size, nb_inputs)
                        this->w_.shape = (nb_inputs, nodes)
                        this->b_.shape = (nodes)
                        this->raw_dw_.shape = (nb_inputs, nodes)
                        this->raw_db_.shape = (nodes)
                        this->final_dw_.shape = (nb_inputs, nodes)
                        this->final_db_.shape = (nodes)
                    */

                    uint_fast64_t batch_size = deriv.shape().front();

                    // dA[l-1]
                    deriv.matmul(this->w_.transpose(), prev_deriv_output); // dA[l-1] = dZ[l] * W[l]
                    
                    // dW[l]
                    cached_input.transpose().matmul(deriv, this->raw_dw_); // dW[l] = A[l-1] * dZ[l]
                    this->raw_dw_.divide_eq(batch_size); // dW[l] /= batch_size (* (1 / m));
                    
                    // db[l]
                    deriv.sum({0}, this->raw_db_); // db[l] = sum of dZ (axis = 0)
                    // since (dZ / db) = 1, (dL / db) = (dL / dZ) * 1 = (dL / dZ)
                    this->raw_db_.divide_eq(batch_size); // db[l] /= batch_size

                    // *** implement - weights update ***
                }

                void build(const Shape &input_shape) override final {
                    // input_shape = (inputs)
                    assert(input_shape.size() == 1);

                    this->w_ = NDArray<T>({input_shape.front(), this->nodes_}, -1, 1);
                    this->b_ = NDArray<T>({this->nodes_}, 0);
                    this->raw_dw_ = NDArray<T>(this->w_.shape());
                    this->raw_db_ = NDArray<T>(this->b_.shape());
                    this->final_dw_ = NDArray<T>(this->w_.shape());
                    this->final_db_ = NDArray<T>(this->b_.shape());

                    this->output_shape_ = {this->nodes_};
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }

                uint_fast64_t params() const noexcept override final {
                    return this->w_.size() + this->b_.size();
                }
                
                inline const NDArray<T>& w() const noexcept {
                    return this->w_;
                }

                inline const NDArray<T>& b() const noexcept {
                    return this->b_;
                }

                inline const NDArray<T>& raw_dw() const noexcept {
                    return this->raw_dw_;
                }

                inline const NDArray<T>& raw_db() const noexcept {
                    return this->raw_db_;
                }

                inline const NDArray<T>& final_dw() const noexcept {
                    return this->final_dw_;
                }

                inline const NDArray<T>& final_db() const noexcept {
                    return this->final_db_;
                }
        };
    }

    using namespace impl;
}




#endif