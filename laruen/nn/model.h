
#ifndef LARUEN_NN_MODEL_H_
#define LARUEN_NN_MODEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <tuple>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/layers/layer.h"
#include "laruen/nn/optimizers/optimizer.h"
#include "laruen/nn/losses/loss.h"
#include "laruen/nn/metrics/metric.h"
#include "laruen/nn/utils/utils.h"
#include "laruen/nn/utils/data_view.h"
#include "laruen/nn/utils/verbose_settings.h"
#include "laruen/nn/history.h"

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
        using laruen::nn::utils::DataView;
        using laruen::nn::utils::VerboseSettings;

        template <typename T = float32_t>
        class Model {
            private:
                std::vector<std::shared_ptr<Layer<T>>> layers_;
                std::vector<NDArray<T>> batch_outputs_;
                std::vector<NDArray<T>> batch_grads_;
                std::vector<NDArray<T>> remaining_train_outputs_;
                std::vector<NDArray<T>> remaining_train_grads_;
                NDArray<T> input_batch_grad_;
                NDArray<T> input_remaining_train_grad_;
                std::shared_ptr<Loss<T>> loss_;
                std::shared_ptr<Optimizer<T>> optimizer_;
                std::vector<std::shared_ptr<Metric<T>>> metrics_;
                bool built_;
                bool compiled_;
                VerboseSettings verbose_settings;

            public:
                Model(const std::vector<std::shared_ptr<Layer<T>>> &layers)
                : layers_(layers), batch_outputs_(layers.size()), batch_grads_(layers.size()),
                remaining_train_outputs_(layers.size()), remaining_train_grads_(layers.size()),
                built_(false), compiled_(false)
                {}
                
                void build(Shape::const_iterator begin, Shape::const_iterator end) {
                    // input_shape = (dim1, ...)
                    this->layers_.front()->build(begin, end);

                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->build(this->layers_[i - 1]->output_shape());
                    }

                    this->built_ = true;
                }

                inline void build(const Shape &input_shape) {
                    this->build(input_shape.cbegin(), input_shape.cend());
                }

                void compile(const std::shared_ptr<Optimizer<T>> &optimizer, const std::shared_ptr<Loss<T>> &loss,
                const std::vector<std::shared_ptr<Metric<T>>> &metrics)
                {
                    this->optimizer_ = optimizer;
                    this->loss_ = loss;
                    this->metrics_.resize(metrics.size() + 1);
                    this->metrics_.front() = loss;
                    std::copy(metrics.cbegin(), metrics.cend(), this->metrics_.begin() + 1);

                    uint_fast64_t required_caches = this->optimizer_->required_caches();

                    for(auto layer = this->layers_.begin();layer != this->layers_.end();layer++) {
                        (*layer)->compile(required_caches);
                    }

                    this->compiled_ = true;
                }

                inline void compile(const std::shared_ptr<Optimizer<T>> &optimizer, const std::shared_ptr<Loss<T>> &loss) {
                    this->compile(optimizer, loss, {});
                }

                void summary() const {
                    constexpr uint_fast64_t sec1_len = 18;
                    constexpr uint_fast64_t sec2_len = 20;
                    constexpr uint_fast64_t all_len = 44;

                    auto shape_str = [](auto shape) noexcept -> std::string {
                        std::string str("(None, ");
                        auto iter = shape.cbegin();
                        for(;iter != shape.cend() - 1;iter++) {
                            str += std::to_string(*iter) + ", ";
                        }
                        str += std::to_string(*iter) + ')';
                        return str;
                    };

                    if(!this->built_) {
                        throw std::logic_error("model must be built before calling summary");
                    }

                    uint_fast64_t total_params = 0;
                    uint_fast64_t curr_params;
                    std::string shape_string;

                    std::string str("Layer Type        Output Shape        Params\n");
                    str.append(all_len, '-');
                    str.push_back('\n');

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        str.append(this->layers_[i]->name());
                        str.append(sec1_len - std::strlen(this->layers_[i]->name()), ' ');
                        str.append(shape_string = shape_str(this->layers_[i]->output_shape()));
                        str.append(sec2_len - shape_string.size(), ' ');
                        str.append(std::to_string(curr_params = this->layers_[i]->params()));
                        str.push_back('\n');
                        total_params += curr_params;
                    }

                    str.append("\nTotal Params: ");
                    str.append(std::to_string(total_params));
                    str.push_back('\n');
                    str.append(all_len, '-');

                    std::cout << str << std::endl;
                }

                inline void forward(const NDArray<T> &input, std::vector<NDArray<T>> &outputs) {
                    this->layers_.front()->forward(input, outputs.front());
                    
                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->forward(outputs[i - 1], outputs[i]);
                    }
                }

                inline void backward(const NDArray<T> &input, const NDArray<T> &y_true,
                const std::vector<NDArray<T>> &outputs, std::vector<NDArray<T>> &grads, NDArray<T> &input_grad)
                {
                    this->loss_->backward(y_true, outputs.back(), grads.back());

                    for(uint_fast64_t i = grads.size();i-- > 1;) {
                        this->layers_[i]->backward(grads[i], outputs[i - 1], outputs[i], grads[i - 1]);
                        this->layers_[i]->update_weights(*this->optimizer_);
                    }

                    this->layers_.front()->backward(grads.front(), input, outputs.front(), input_grad);
                    this->layers_.front()->update_weights(*this->optimizer_);
                }

                inline void train_batch(const NDArray<T> &input, const NDArray<T> &y_true,
                std::vector<NDArray<T>> &outputs, std::vector<NDArray<T>> &grads, NDArray<T> &input_grad,
                bool update_optimizer_params = false)
                {
                    this->forward(input, outputs);
                    this->backward(input, y_true, outputs, grads, input_grad);

                    if(update_optimizer_params) {
                        this->optimizer_->update_optimizer_params();
                    }
                }

                History<T> fit(const NDArray<T> &x, const NDArray<T> &y, uint_fast64_t batch_size = 32,
                uint_fast64_t epochs = 1, bool verbose = true)
                {
                    using laruen::nn::utils::batch_view;

                    if(!this->built_ || !this->compiled_) {
                        throw std::logic_error("model must be built & compiled before calling fit");
                    }

                    DataView<T> train_view(x, y, batch_size);

                    uint_fast64_t epoch;
                    uint_fast64_t batch;

                    bool last; // used only when verbose = true

                    this->reset_metrics(this->metrics_, epochs);

                    if(train_view.full_batches) {
                        this->construct(this->batch_outputs_, this->batch_grads_,
                        this->input_batch_grad_, train_view.x_batch.shape());
                    }

                    if(train_view.remaining) {
                        this->construct(this->remaining_train_outputs_, this->remaining_train_grads_,
                        this->input_remaining_train_grad_, train_view.x_remaining.shape());
                    }

                    for(epoch = 0;epoch < epochs;epoch++) {
                        for(batch = 0;batch < train_view.full_batches;batch++) {
                            this->train_batch(train_view.x_batch, train_view.y_batch, this->batch_outputs_,
                            this->batch_grads_, this->input_batch_grad_, true);

                            this->compute_metrics(this->metrics_, train_view.y_batch, this->batch_outputs_.back(), 1.0);

                            if(verbose && ((last = !train_view.remaining &&
                            (batch == train_view.full_batches - 1)) || !(batch % this->verbose_settings.rate)))
                            {
                                this->verbose(epoch, epochs, (T)(batch + 1), batch, train_view.batches, last);
                            }

                            train_view.next_batch();
                        }

                        if(train_view.remaining) {
                            this->train_batch(train_view.x_remaining, train_view.y_remaining, this->remaining_train_outputs_,
                            this->remaining_train_grads_, this->input_remaining_train_grad_, true);

                            this->compute_metrics(this->metrics_, train_view.y_remaining,
                            this->remaining_train_outputs_.back(), train_view.remaining_ratio);

                            if(verbose) {
                                this->verbose(epoch, epochs, train_view.partial_batches, batch, train_view.batches, true);
                            }
                        }

                        this->average_metrics(this->metrics_, train_view.partial_batches);


                        train_view.reset_batch_views(x.data(), y.data());
                    }

                    return History(this->metrics_);
                }

                History<T> evaluate(const NDArray<T> &x, const NDArray<T> &y,
                uint_fast64_t batch_size = 32, bool verbose = true)
                {
                    using laruen::nn::utils::batch_view;

                    if(!this->built_ || !this->compiled_) {
                        throw std::logic_error("model must be built & compiled before calling evaluate");
                    }

                    DataView<T> data_view(x, y, batch_size);

                    uint_fast64_t batch;
                    bool last; // used only when verbose = true
                    
                    this->reset_metrics(this->metrics_, 1);

                    if(data_view.full_batches) {
                        this->construct_forward(this->batch_outputs_, data_view.x_batch.shape().front());
                    }

                    if(data_view.remaining) {
                        this->construct_forward(this->remaining_train_outputs_, data_view.x_remaining.shape().front());
                    }

                    for(batch = 0;batch < data_view.full_batches;batch++) {
                        this->forward(data_view.x_batch, this->batch_outputs_);

                        this->compute_metrics(this->metrics_, data_view.y_batch, this->batch_outputs_.back(), 1.0);

                        if(verbose && ((last = !data_view.remaining &&
                        (batch == data_view.full_batches - 1)) || !(batch % this->verbose_settings.rate)))
                        {
                            this->verbose(0, 1, (T)(batch + 1), batch, data_view.batches, last);
                        }

                        data_view.next_batch();
                    }

                    if(data_view.remaining) {
                        this->forward(data_view.x_remaining, this->remaining_train_outputs_);

                        this->compute_metrics(this->metrics_, data_view.y_remaining,
                        this->remaining_train_outputs_.back(), data_view.remaining_ratio);

                        if(verbose) {
                            this->verbose(0, 1, data_view.partial_batches, batch, data_view.batches, true);
                        }
                    }

                    this->average_metrics(this->metrics_, data_view.partial_batches);

                    return History(this->metrics_);
                }

                NDArray<T>& predict(const NDArray<T> &x) {
                    this->construct_forward(this->batch_outputs_, x.shape().front());

                    this->forward(x, this->batch_outputs_);

                    return this->batch_outputs_.back();
                }

                void load_weights(std::ifstream &file) {
                    // file must be opened in binary mode
                    if(!this->built_) {
                        throw std::logic_error("model must be built before calling load_weights");
                    }

                    for(auto layer = this->layers_.cbegin();layer != this->layers_.cend();layer++) {
                        (*layer)->load_weights(file);
                    }
                }

                inline void load_weights(const std::string &filepath) {
                    std::ifstream file(filepath, std::ios::binary);
                    this->load_weights(file);
                }

                void save_weights(std::ofstream &file) const {
                    // file must be opened in binary mode
                    for(auto layer = this->layers_.cbegin();layer != this->layers_.cend();layer++) {
                        (*layer)->save_weights(file);
                    }
                }

                inline void save_weights(const std::string &filepath) const {
                    std::ofstream file(filepath, std::ios::binary);
                    this->save_weights(file);
                }

                inline auto& layers() noexcept {
                    return this->layers_;
                }

                inline std::vector<NDArray<T>>& batch_outputs() noexcept {
                    return this->batch_outputs_;
                }

                inline std::vector<NDArray<T>>& batch_grads() noexcept {
                    return this->batch_grads_;
                }

                inline std::vector<NDArray<T>>& remaining_train_outputs() noexcept {
                    return this->remaining_train_outputs_;
                }

                inline std::vector<NDArray<T>>& remaining_train_grads() noexcept {
                    return this->remaining_train_grads_;
                }

                inline NDArray<T>& input_batch_grad() noexcept {
                    return this->input_batch_grad_;
                }

                inline NDArray<T>& input_remaining_train_grad() noexcept {
                    return this->input_remaining_train_grad_;
                }

                inline std::shared_ptr<Loss<T>>& loss() noexcept {
                    return this->loss_;
                }

                inline std::shared_ptr<Optimizer<T>>& optimizer() noexcept {
                    return this->optimizer_;
                }

                inline auto& metrics() noexcept {
                    return this->metrics_;
                }
                
                inline void reset_metrics(std::vector<std::shared_ptr<Metric<T>>> &metrics, uint_fast64_t new_size) {
                    for(auto metric = metrics.begin();metric != metrics.end();metric++) {
                        (*metric)->values().assign(new_size, 0);
                    }
                }

                inline void compute_metrics(std::vector<std::shared_ptr<Metric<T>>> &metrics,
                const NDArray<T> &y_true, const NDArray<T> &y_pred, T size_ratio = 1)
                {
                    for(auto metric = metrics.begin();metric != metrics.end();metric++) {
                        (*metric)->values().front() += size_ratio * (**metric)(y_true, y_pred);
                    }
                }

                inline void average_metrics(std::vector<std::shared_ptr<Metric<T>>> &metrics,
                T partial_batches)
                {
                    for(auto metric = metrics.begin();metric != metrics.end();metric++) {
                        (*metric)->values().front() /= partial_batches;
                    }
                }

                void free_resources() noexcept {
                    this->batch_outputs_.clear();
                    this->batch_grads_.clear();
                    this->remaining_train_outputs_.clear();
                    this->remaining_train_grads_.clear();
                    this->input_batch_grad_.free();
                    this->input_remaining_train_grad_.free();
                }
                        
            private:
                void construct_forward(std::vector<NDArray<T>> &batch_outputs, uint_fast64_t batch_size) noexcept
                {
                    using laruen::nn::utils::add_batch_shape;
                    
                    if(batch_outputs.front().shape().size() && batch_outputs.front().shape().front() == batch_size) {
                        return;
                    }

                    batch_outputs.resize(this->layers_.size());

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        batch_outputs[i].resize(add_batch_shape(this->layers_[i]->output_shape(), batch_size));
                    }
                }

                void construct_backward(const std::vector<NDArray<T>> &batch_outputs, std::vector<NDArray<T>> &batch_grads,
                NDArray<T> &input_grad, const Shape &input_batch_shape) noexcept
                {
                    if(batch_grads.front().shape().size() &&
                    batch_grads.front().shape().front() == input_batch_shape.front())
                    {
                        return;
                    }

                    input_grad.resize(input_batch_shape);

                    batch_grads.resize(this->layers_.size());

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        batch_grads[i].resize(batch_outputs[i].shape());
                    }
                }

                inline void construct(std::vector<NDArray<T>> &batch_outputs, std::vector<NDArray<T>> &batch_grads,
                NDArray<T> &input_grad, const Shape &input_batch_shape) noexcept
                {
                    this->construct_forward(batch_outputs, input_batch_shape.front());
                    this->construct_backward(batch_outputs, batch_grads, input_grad, input_batch_shape);
                }

                void verbose(uint_fast64_t epoch_index, uint_fast64_t epochs,
                T partial_batch, uint_fast64_t batch_index, uint_fast64_t batches, bool last)
                {
                    // *** written very badly ***
                    static std::string str;
                    static uint_fast64_t max_str_length = 0;

                    uint_fast64_t batch_nb = batch_index + 1;
                    uint_fast64_t epoch_nb = epoch_index + 1;

                    uint_fast16_t progress = (uint_fast16_t)std::round(((T)(batch_nb - (batch_nb < batches)))
                    * this->verbose_settings.progress_bar_length / batches);

                    uint_fast64_t remaining_bar_len = std::max((int_fast32_t)
                    (this->verbose_settings.progress_bar_length - (progress + 1)), (int_fast32_t)0);

                    str.clear();

                    str.append("epoch ");
                    str.append(std::to_string(epoch_nb));
                    str.push_back('/');
                    str.append(std::to_string(epochs));
                    str.append(" - ");
                    str.append(std::to_string(batch_nb));
                    str.push_back('/');
                    str.append(std::to_string(batches));
                    str.append(" - [");
                    str.append(progress, '=');

                    if(batch_nb < batches) {
                        str.push_back('>');
                    }

                    str.append(remaining_bar_len, ' ');
                    str.push_back(']');

                    for(auto metric = this->metrics_.cbegin();metric != this->metrics_.cend();metric++) {
                        std::string metric_str = std::to_string((*metric)->values()[epoch_index] / partial_batch);

                        str.append(" - ");
                        str.append((*metric)->name());
                        str.push_back(':');
                        str.push_back(' ');
                        str.append(metric_str.cbegin(), metric_str.cbegin() + metric_str.find('.') + this->verbose_settings.precision + 1);
                    }

                    if(str.size() < max_str_length) {
                        str.append(max_str_length - str.size(), ' ');
                    }
                    else {
                        max_str_length = str.size();
                    }

                    if(last) {
                        std::cout << str << std::endl;
                    }
                    else {
                        std::cout << str << '\r' << std::flush;
                    }
                }
        };

        template <typename T> Model(std::vector<std::shared_ptr<Layer<T>>>&) -> Model<T>;
    }

    using namespace impl;
}


#endif