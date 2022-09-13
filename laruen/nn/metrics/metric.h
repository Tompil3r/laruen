
#ifndef LARUEN_NN_METRICS_METRIC_H_
#define LARUEN_NN_METRICS_METRIC_H_

#include <vector>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"

namespace laruen::nn::metrics {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Metric {
            private:
                std::vector<T> values_;
                std::string name_;

            public:
                virtual ~Metric()
                {}

                inline Metric(const std::string &name) : name_(name)
                {}

                inline std::vector<T>& values() noexcept {
                    return this->values_;
                }

                inline const std::vector<T>& values() const noexcept {
                    return this->values_;
                }

                inline const std::string& name() const noexcept {
                    return this->name_;    
                }

                inline void name(const std::string &name) {
                    this->name_ = name;    
                }

                virtual Metric<T>* clone() const = 0;

                virtual T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const = 0;                
        };
    }

    using namespace impl;
}

#endif