
#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/layers/layer.h"
#include "src/nn/optimizers/optimizer.h"
#include "src/nn/losses/loss.h"
#include "src/nn/utils.h"

namespace laruen::nn {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::float32_t;
        using laruen::nn::layers::Layer;
        using laruen::nn::optimizers::Optimizer;
        using laruen::nn::losses::Loss;

        template <typename T = float32_t>
        class Model {
            private:
                std::vector<Layer<T>*> layers_;
                std::vector<NDArray<T>> batch_outputs_;
                std::vector<NDArray<T>> batch_derivs_;
                std::vector<NDArray<T>> remaining_outputs_;
                std::vector<NDArray<T>> remaining_derivs_;
                NDArray<T> input_batch_deriv;
                NDArray<T> input_remaining_deriv;
                Loss<T> *loss_;
                Optimizer<T> *optimizer_;
                uint_fast64_t batch_size_;
                uint_fast64_t remaining_size_;
                bool manage_resources_;
                
            public:
                ~Model() {
                    if(!this->manage_resources_) {
                        return;
                    }

                    delete this->loss_;
                    delete this->optimizer_;

                    for(auto iter = this->layers_.begin();iter != this->layers_.end();iter++) {
                        delete *iter;
                    }
                }

                Model(const std::vector<Layer<T>*> &layers, bool manage_resources = true)
                : layers_(layers), batch_outputs_(layers.size()), batch_derivs_(layers.size()),
                remaining_outputs_(layers.size()), remaining_derivs_(layers.size()),
                loss_(nullptr), optimizer_(nullptr), batch_size_(0),
                remaining_size_(0), manage_resources_(manage_resources)
                {}
                
                void build(Shape::const_iterator begin, Shape::const_iterator end) {
                    // input_shape = (dim1, ...)
                    this->layers_[0]->build(begin, end);

                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->build(this->layers_[i - 1]->output_shape());
                    }
                }

                inline void build(const Shape &input_shape) {
                    this->build(input_shape.cbegin(), input_shape.cend());
                }

                void compile(Optimizer<T> *optimizer, Loss<T> *loss) {
                    this->optimizer_ = optimizer;
                    this->loss_ = loss;

                    uint_fast64_t required_caches = this->optimizer_->required_caches();

                    for(auto layer = this->layers_.begin();layer != this->layers_.end();layer++) {
                        (*layer)->compile(required_caches);
                    }
                }

                std::string summary() const noexcept {
                    std::string str("Layer Name\t\tOutput Shape\t\tParams\n");
                    str += std::string(60, '-');
                    str += '\n';

                    auto shape_str = [](auto shape) noexcept -> std::string {
                        std::string str("(None, ");
                        auto iter = shape.cbegin();
                        for(;iter != shape.cend() - 1;iter++) {
                            str += std::to_string(*iter) + ", ";
                        }
                        str += std::to_string(*iter) + ')';
                        return str;
                    };

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        str += this->layers_[i]->name();
                        str += "\t\t\t";
                        str += shape_str(this->layers_[i]->output_shape());
                        str += "\t\t";
                        str += std::to_string(this->layers_[i]->params());
                        str += '\n';
                    }
                    
                    return str;
                }

                inline void forward(const NDArray<T> &input, std::vector<NDArray<T>> &outputs) {
                    this->layers_.front()->forward(input, outputs.front());
                    
                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->forward(outputs[i - 1], outputs[i]);
                    }
                }

                inline void backward(const NDArray<T> &input, const NDArray<T> &y_true,
                const std::vector<NDArray<T>> &outputs, std::vector<NDArray<T>> &derivs, NDArray<T> &input_deriv)
                {
                    this->loss_->backward(y_true, outputs.back(), derivs.back());

                    for(uint_fast64_t i = derivs.size();i-- > 1;) {
                        this->layers_[i]->backward(derivs[i], outputs[i - 1], outputs[i], derivs[i - 1]);
                        this->layers_[i]->update_weights(*this->optimizer_);
                    }

                    this->layers_.front()->backward(derivs.front(), input, outputs.front(), input_deriv);
                    this->layers_.front()->update_weights(*this->optimizer_);
                }

                inline void train_batch(const NDArray<T> &input, const NDArray<T> &y_true,
                std::vector<NDArray<T>> &outputs, std::vector<NDArray<T>> &derivs, NDArray<T> &input_deriv,
                bool update_optimizer_params = false)
                {
                    this->forward(input, outputs);
                    this->backward(input, y_true, outputs, derivs, input_deriv);

                    if(update_optimizer_params) {
                        this->optimizer_->update_optimizer_params();
                    }
                }

                inline const std::vector<Layer<T>*>& layers() const noexcept {
                    return this->layers_;
                }

                inline const std::vector<NDArray<T>>& batch_outputs() const noexcept {
                    return this->batch_outputs_;
                }
                        
            private:
                void construct(std::vector<NDArray<T>> &batch_outputs, std::vector<NDArray<T>> &batch_derivs,
                uint_fast64_t batch_size) noexcept
                {
                    using laruen::nn::utils::batch_shape;

                    batch_outputs.resize(this->layers_.size());
                    batch_derivs.resize(this->layers_.size());

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        batch_outputs[i] = NDArray<T>(batch_shape(this->layers_[i]->output_shape(), batch_size));
                        batch_derivs[i] = NDArray<T>(batch_outputs[i].shape());
                    }
                }
        };
    }

    using namespace impl;
}


#endif