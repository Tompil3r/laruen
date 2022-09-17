
#ifndef LARUEN_NN_METRICS_CATEGORICAL_ACCURACY_H_
#define LARUEN_NN_METRICS_CATEGORICAL_ACCURACY_H_

#include <memory>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/nditer.h"
#include "laruen/nn/metrics/metric.h"

namespace laruen::nn::metrics {

    using laruen::multi::NDArray;
    using laruen::multi::float32_t;
    using laruen::multi::NDIter;

    namespace impl {
        
        template <typename T = float32_t>
        class CategoricalAccuracy : public Metric<T> {            
            public:
                inline CategoricalAccuracy() noexcept
                : Metric<T>("categorical_accuracy")
                {}

                inline CategoricalAccuracy(const std::string &name)
                : Metric<T>(name)
                {}

                inline Metric<T>* clone() const override final {
                    return new CategoricalAccuracy<T>(*this);
                }

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);

                    T score = 0;                    
                    uint_fast64_t nb_samples = y_pred.size() / y_pred.shape().back();
                    uint_fast8_t y_true_iteration_axis = y_true.ndim() - 2;

                    for(uint_fast64_t i = 0;i < nb_samples;i++) {
                        T max = pred_iter.next();
                        uint_fast64_t index_max = 0;

                        for(uint_fast64_t j = 1;j < y_pred.shape().back();j++) {
                            if(pred_iter.current() > max) {
                                max = pred_iter.current();
                                index_max = j;
                            }
                            pred_iter.next();
                        }
                        score += true_iter.ptr[index_max * y_true.strides().back()];
                        true_iter.next(y_true_iteration_axis);
                    }

                    return (score / nb_samples);
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> categorical_accuracy() noexcept {
            return std::shared_ptr<Metric<T>>(new CategoricalAccuracy<T>());
        }
        
        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> categorical_accuracy(const std::string &name) {
            return std::shared_ptr<Metric<T>>(new CategoricalAccuracy<T>(name));
        }
    }

    using namespace impl;
}



#endif