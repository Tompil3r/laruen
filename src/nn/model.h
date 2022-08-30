
#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <vector>
#include <string>
#include <algorithm>
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
                std::vector<NDArray<T>> batch_deriv_;
                std::vector<NDArray<T>> remaining_outputs_;
                std::vector<NDArray<T>> remaining_deriv_;
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
                : layers_(layers), loss_(nullptr), optimizer_(nullptr), batch_size_(0),
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

                void forward(const NDArray<T> &input, std::vector<NDArray<T>> &outputs) {
                    this->layers_[0]->forward(input, outputs[0]);
                    
                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->forward(outputs[i - 1], outputs[i]);
                    }
                }

                inline const std::vector<Layer<T>*>& layers() const noexcept {
                    return this->layers_;
                }

                inline const std::vector<NDArray<T>>& batch_outputs() const noexcept {
                    return this->batch_outputs_;
                }
                        
            private:
                void shape_outputs(std::vector<NDArray<T>> &outputs, std::vector<NDArray<T>> &d_outputs,
                uint_fast64_t batch_size) noexcept
                {
                    using laruen::nn::utils::batch_shape;

                    outputs.resize(this->layers_.size());
                    d_outputs.resize(this->layers_.size());

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        outputs[i] = NDArray<T>(batch_shape(this->layers_[i]->output_shape(), batch_size));
                        d_outputs[i] = NDArray<T>(outputs[i].shape());
                    }
                }
        };
    }

    using namespace impl;
}


#endif