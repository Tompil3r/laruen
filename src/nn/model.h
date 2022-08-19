
#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/layers/layer.h"

namespace laruen::nn {

    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::Shape;
        using laruen::ndlib::float32_t;
        using laruen::nn::layers::Layer;

        template <typename T = float32_t>
        class Model {
            private:
                std::vector<Layer<T>*> layers_;
                std::vector<NDArray<T>> batch_outputs_;
                std::vector<NDArray<T>> batch_d_outputs_;
                std::vector<NDArray<T>> rem_outputs_;
                std::vector<NDArray<T>> rem_d_outputs_;
                uint_fast64_t batch_size_;
                bool manage_resources_;
                
            public:
                ~Model() {
                    if(!this->manage_resources_) {
                        return;
                    }

                    for(auto iter = this->layers_.begin();iter != this->layers_.end();iter++) {
                        delete *iter;
                    }
                }

                Model(std::vector<Layer<T>*> &layers, bool manage_resources = true)
                : layers_(layers), batch_outputs_(layers.size()), batch_d_outputs_(layers.size()),
                batch_size_(0), manage_resources_(manage_resources)
                {}
                
                void build(const Shape &input_shape) {
                    // input_shape = (dim0, ...)
                    this->layers_[0]->build(input_shape);

                    for(uint_fast64_t i = 1;i < this->layers_.size();i++) {
                        this->layers_[i]->build(this->layers_[i - 1]->output_shape());
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
        };
    }

    using namespace impl;
}


#endif