
#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/layers/layer.h"
#include "src/nn/optimizers/optimizer.h"
#include "src/nn/losses/loss.h"
#include "src/nn/metrics/metric.h"
#include "src/nn/utils.h"

namespace laruen::nn {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::Shape;
        using laruen::multi::Strides;
        using laruen::multi::float32_t;
        using laruen::nn::layers::Layer;
        using laruen::nn::optimizers::Optimizer;
        using laruen::nn::losses::Loss;
        using laruen::nn::metrics::Metric;

        template <typename T = float32_t>
        class Model {
            private:
                std::vector<std::shared_ptr<Layer<T>>> layers_;
                std::vector<NDArray<T>> batch_outputs_;
                std::vector<NDArray<T>> batch_derivs_;
                std::vector<NDArray<T>> remaining_outputs_;
                std::vector<NDArray<T>> remaining_derivs_;
                NDArray<T> input_batch_deriv_;
                NDArray<T> input_remaining_deriv_;
                std::shared_ptr<Loss<T>> loss_;
                std::shared_ptr<Optimizer<T>> optimizer_;
                uint_fast64_t batch_size_;
                uint_fast64_t remaining_size_;
                
            public:
                Model(const std::vector<std::shared_ptr<Layer<T>>> &layers)
                : layers_(layers), batch_outputs_(layers.size()), batch_derivs_(layers.size()),
                remaining_outputs_(layers.size()), remaining_derivs_(layers.size()),
                batch_size_(0), remaining_size_(0)
                {}
                
                void build(Shape::const_iterator begin, Shape::const_iterator end) {
                    // input_shape = (dim1, ...)
                    this->layers_.front()->build(begin, end);

                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->build(this->layers_[i - 1]->output_shape());
                    }
                }

                inline void build(const Shape &input_shape) {
                    this->build(input_shape.cbegin(), input_shape.cend());
                }

                void compile(const std::shared_ptr<Optimizer<T>> &optimizer, const std::shared_ptr<Loss<T>> &loss) {
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

                void fit(const NDArray<T> &x, const NDArray<T> &y, const std::vector<std::shared_ptr<Metric<T>>> &metrics,
                uint_fast64_t batch_size = 32, uint_fast64_t epochs = 1, bool verbose = true)
                {
                    using laruen::nn::utils::batch_view;

                    uint_fast64_t batches = x.shape().front() / batch_size;
                    uint_fast64_t remaining_size = x.shape().front() % batch_size;
                    uint_fast64_t verbose_batches = batches + (remaining_size > 0); // for verbose purposes only

                    const NDArray<T> x_batch_view = batch_view(x, batch_size);
                    const NDArray<T> y_batch_view = batch_view(y, batch_size);

                    const NDArray<T> x_remaining_view;
                    const NDArray<T> y_remaining_view;

                    uint_fast64_t x_batch_stride = batch_size * x.strides().front();
                    uint_fast64_t y_batch_stride = batch_size * y.strides().front();

                    if(this->batch_size_ != batch_size) {
                        this->construct(this->batch_outputs_, this->batch_derivs_, this->input_batch_deriv_,
                        x_batch_view.shape(), batch_size);
                        this->batch_size_ = batch_size;
                    }

                    if(remaining_size && this->remaining_size_ != remaining_size) {
                        x_remaining_view = batch_view(x, remaining_size);
                        y_remaining_view = batch_view(y, remaining_size);
                        x_remaining_view.data(x.data() + batches * x.strides().front());
                        y_remaining_view.data(y.data() + batches * y.strides().front());

                        this->construct(this->remaining_outputs_, this->remaining_derivs_, this->input_remaining_deriv_,
                        x_remaining_view.shape(), remaining_size);
                        
                        this->remaining_size_ = remaining_size;                    
                    }

                    uint_fast64_t epoch;
                    uint_fast64_t batch;

                    for(epoch = 1;epoch <= epochs;epoch++) {
                        for(batch = 1;batch <= batches;batch++) {
                            this->train_batch(x_batch_view, y_batch_view, this->batch_outputs_,
                            this->batch_derivs_, this->input_batch_deriv_, true);

                            x_batch_view.data(x_batch_view.data() + x_batch_stride);
                            y_batch_view.data(y_batch_view.data() + y_batch_stride);

                            if(verbose) {
                                this->verbose(epoch, epochs, batch, verbose_batches, y_batch_view, this->batch_outputs_.back(),
                                metrics, !remaining_size && batch == batches - 1);
                            }
                        }

                        if(remaining_size) {
                            this->train_batch(x_remaining_view, y_remaining_view, this->remaining_outputs_,
                            this->remaining_derivs_, this->input_remaining_deriv_, true);

                            if(verbose) {
                                this->verbose(epoch, epochs, batch, verbose_batches, y_remaining_view, this->remaining_outputs_.back(),
                                metrics, true);
                            }
                        }

                        x_batch_view.data(x.data());
                        y_batch_view.data(y.data());
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
                NDArray<T> &input_deriv, const Shape &input_shape, uint_fast64_t batch_size) noexcept
                {
                    using laruen::nn::utils::add_batch_shape;

                    input_deriv = NDArray<T>(input_shape);
                    input_deriv.shape().front() = batch_size;

                    batch_outputs.resize(this->layers_.size());
                    batch_derivs.resize(this->layers_.size());

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        batch_outputs[i] = NDArray<T>(add_batch_shape(this->layers_[i]->output_shape(), batch_size));
                        batch_derivs[i] = NDArray<T>(batch_outputs[i].shape());
                    }
                }

                void verbose(uint_fast64_t epoch, uint_fast64_t epochs, uint_fast64_t batch,
                uint_fast64_t batches, const NDArray<T> &y_true, const NDArray<T> &y_pred,
                const std::vector<std::shared_ptr<Metric<T>>> &metrics, bool last)
                {
                    constexpr uint_fast8_t precision = 3;
                    constexpr uint_fast16_t progress_bar_len = 20;

                    uint_fast16_t progress = (uint_fast16_t)std::round(((T)(batch - 1)) * progress_bar_len / batches);

                    std::cout << "epoch: " << epoch << '/' << epochs << " - [" << std::string(progress, '=') <<
                    '>' << std::string(std::min(batches - (progress + 1), (uint_fast64_t)0), ' ') << "] - loss: "
                    << std::setprecision(precision) << (*this->loss_)(y_true, y_pred);

                    for(auto metric = metrics.cbegin();metric != metrics.cend();metric++) {
                        std::cout << " - " << (*metric)->name() << ": " <<
                        std::setprecision(precision) << (**metric)(y_true, y_pred);
                    }

                    if(last) {
                        std::cout << std::endl;
                    }
                    else {
                        std::cout << '\r';
                    }
                }
        };

        template <typename T> Model(std::vector<std::shared_ptr<Layer<T>>>&) -> Model<T>;
    }

    using namespace impl;
}


#endif