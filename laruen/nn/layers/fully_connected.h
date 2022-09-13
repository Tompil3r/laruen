
#ifndef LARUEN_NN_LAYERS_FULLY_CONNECTED_H_
#define LARUEN_NN_LAYERS_FULLY_CONNECTED_H_

#include <cassert>
#include <utility>
#include <vector>
#include <memory>
#include <cstdint>
#include <fstream>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/layers/layer.h"
#include "laruen/nn/optimizers/optimizer.h"
#include "laruen/nn/initializers/initializer.h"
#include "laruen/nn/initializers/glorot_uniform.h"
#include "laruen/nn/initializers/zeros.h"

namespace laruen::nn::layers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;
        using laruen::nn::optimizers::Optimizer;
        using laruen::nn::initializers::Initializer;
        using laruen::nn::initializers::glorot_uniform;
        using laruen::nn::initializers::zeros;

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
                std::vector<NDArray<T>> opt_dw_caches_;
                std::vector<NDArray<T>> opt_db_caches_;
                uint_fast32_t nodes_;
                std::shared_ptr<Initializer<T>> kernel_initializer_;
                std::shared_ptr<Initializer<T>> bias_initializer_;

            public:
                FullyConnected(uint_fast32_t nodes,
                std::shared_ptr<Initializer<T>> kernel_initializer = glorot_uniform(),
                std::shared_ptr<Initializer<T>> bias_initializer = zeros()) noexcept
                : nodes_(nodes), kernel_initializer_(kernel_initializer), bias_initializer_(bias_initializer)
                {}

                NDArray<T>& forward(const NDArray<T> &input, NDArray<T> &output) const override final {
                    // input.shape = (number samples, inputs)
                    // output.shape = (number samples, nodes)
                    input.matmul(this->w_, output);
                    output.add_eq(this->b_);
                    
                    return output;
                }

                NDArray<T> forward(const NDArray<T> &input) override final {
                    assert(input.ndim() == 2);

                    if(!this->w_.size() || (input.shape().back() != this->w_.shape().front())) {
                        this->build(input.shape().cbegin() + 1, input.shape().cend());
                    }

                    NDArray<T> output(Shape{input.shape().front(), this->nodes_});
                    this->forward(input, output);

                    return output;
                }

                /**
                 * @brief calculates the gradient of the Loss function with respect to A
                 * @param grad dZ (dL / dZ) (dL = dLoss)
                 * @param cached_input A
                 * @param cached_output Z (A * W) (matmul)
                 * @param prev_grad_output dA (dL / dA)
                 */
                void backward(const NDArray<T> &grad, const NDArray<T> &cached_input,
                const NDArray<T> &cached_output, NDArray<T> &prev_grad_output) noexcept override final
                {
                    /*
                        grad.shape = (batch_size, nodes)
                        cached_input.shape = (batch_size, nb_inputs)
                        cached_output.shape = (batch_size, nodes)
                        prev_grad_output.shape = (batch_size, nb_inputs)
                        this->w_.shape = (nb_inputs, nodes)
                        this->b_.shape = (nodes)
                        this->raw_dw_.shape = (nb_inputs, nodes)
                        this->raw_db_.shape = (nodes)
                        this->final_dw_.shape = (nb_inputs, nodes)
                        this->final_db_.shape = (nodes)
                    */

                    uint_fast64_t batch_size = grad.shape().front();

                    // dA[l-1]
                    grad.matmul(this->w_.transpose(), prev_grad_output); // dA[l-1] = dZ[l] * W[l]
                    
                    // dW[l]
                    cached_input.transpose().matmul(grad, this->raw_dw_); // dW[l] = A[l-1] * dZ[l]
                    this->raw_dw_.divide_eq(batch_size); // dW[l] /= batch_size (* (1 / m));
                    
                    // db[l]
                    grad.sum({0}, this->raw_db_); // db[l] = sum of dZ (axis = 0)
                    // since (dZ / db) = 1, (dL / db) = (dL / dZ) * 1 = (dL / dZ)
                    this->raw_db_.divide_eq(batch_size); // db[l] /= batch_size
                }

                inline void update_weights(const Optimizer<T> &optimizer) override final {
                    optimizer.update_weights(this->w_, this->raw_dw_, this->final_dw_, this->opt_dw_caches_);
                    optimizer.update_weights(this->b_, this->raw_db_, this->final_db_, this->opt_db_caches_);
                }

                void build(Shape::const_iterator begin, Shape::const_iterator end) override final {
                    // input shape = (inputs)
                    assert((Shape::size_type)(end - begin) == 1);

                    this->w_ = laruen::multi::empty(Shape{*begin, this->nodes_});
                    this->b_ = laruen::multi::empty(Shape{this->nodes_});

                    (*this->kernel_initializer_)(*begin, this->nodes_, this->w_);
                    (*this->bias_initializer_)(*begin, this->nodes_, this->b_);

                    this->raw_dw_ = NDArray<T>(this->w_.shape());
                    this->raw_db_ = NDArray<T>(this->b_.shape());
                    this->final_dw_ = NDArray<T>(this->w_.shape());
                    this->final_db_ = NDArray<T>(this->b_.shape());

                    this->output_shape_ = {this->nodes_};
                }

                inline void build(const Shape &input_shape) override final {
                    this->build(input_shape.cbegin(), input_shape.cend());
                }

                void compile(uint_fast64_t required_caches) override final {
                    this->opt_dw_caches_.resize(required_caches);
                    this->opt_db_caches_.resize(required_caches);

                    for(uint_fast64_t i = 0;i < required_caches;i++) {
                        this->opt_dw_caches_[i] = NDArray<T>(this->raw_dw_.shape(), 0);
                        this->opt_db_caches_[i] = NDArray<T>(this->raw_db_.shape(), 0);
                    }
                }

                inline void save_weights(std::ofstream &file, int_fast64_t offset = 0) const override final {
                    this->w_.save_buffer(file, offset);
                    this->b_.save_buffer(file);
                }
                
                inline void load_weights(std::ifstream &file, int_fast64_t offset = 0) override final {
                    this->w_.template load_buffer<T>(file, offset);
                    this->b_.template load_buffer<T>(file);
                }

                const char* name() const noexcept override final {
                    return "Fully Connected";
                }

                uint_fast64_t params() const noexcept override final {
                    return this->w_.size() + this->b_.size();
                }
                
                inline const NDArray<T>& w() const noexcept {
                    return this->w_;
                }

                inline NDArray<T>& w() noexcept {
                    return this->w_;
                }

                inline const NDArray<T>& b() const noexcept {
                    return this->b_;
                }

                inline NDArray<T>& b() noexcept {
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

        template <typename T = float32_t>
        inline std::shared_ptr<Layer<T>> fully_connected(uint_fast32_t nodes,
        std::shared_ptr<Initializer<T>> kernel_initializer = glorot_uniform(),
        std::shared_ptr<Initializer<T>> bias_initializer = zeros()) noexcept
        {
            return std::shared_ptr<Layer<T>>(new FullyConnected<T>(nodes, kernel_initializer, bias_initializer));
        }
    }

    using namespace impl;
}




#endif