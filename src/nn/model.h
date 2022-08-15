
#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <vector>
#include <string>
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
                std::vector<NDArray<T>> outputs_;
                uint_fast64_t batch_size_;
                bool manage_layers_;
                
            public:
                ~Model() {
                    if(this->manage_layers_) {
                        for(auto iter = this->layers_.begin();iter != this->layers_.end();iter++) {
                            delete *iter;
                        }
                    }
                }

                Model(std::vector<Layer<T>*> &layers, bool manage_layers = true)
                : layers_(layers), outputs_(layers.size()),
                batch_size_(0), manage_layers_(manage_layers)
                {}
                
                void build(const Shape &input_shape) {
                    // input_shape = (number of samples, dim1, ...)
                    Shape next_shape = input_shape;

                    for(uint_fast64_t i = 0;i < this->layers_.size();i++) {
                        next_shape = this->layers_[i]->build(next_shape);
                        this->outputs_[i] = NDArray<T>(next_shape);
                    }
                }

                std::string summary() const noexcept {
                    std::string str("Layer Name\t\tOutput Shape\n");
                    str += std::string(36, '-');
                    str += '\n';

                    auto shape_str = [](auto shape) noexcept -> std::string {
                        std::string str("(");
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
                        str += shape_str(this->outputs_[i].shape());
                        str += '\n';
                    }
                    
                    return str;
                }
        };
    }

    using namespace impl;
}


#endif