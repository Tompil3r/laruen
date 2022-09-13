
#ifndef LARUEN_NN_METRICS_SPARSE_CATEGORICAL_ACCURACY_H_
#define LARUEN_NN_METRICS_SPARSE_CATEGORICAL_ACCURACY_H_

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
        class SparseCategoricalAccuracy : public Metric<T> {            
            public:
                inline SparseCategoricalAccuracy() noexcept
                : Metric<T>("sparse categorical accuracy")
                {}

                inline SparseCategoricalAccuracy(const std::string &name)
                : Metric<T>(name)
                {}

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);

                    T score = 0;

                    for(uint_fast64_t i = 0;i < y_true.size();i++) {
                        T max = pred_iter.next();
                        uint_fast64_t index_max = 0;

                        for(uint_fast64_t j = 1;j < y_pred.shape().back();j++) {
                            if(pred_iter.current() > max) {
                                max = pred_iter.current();
                                index_max = j;
                            }
                            pred_iter.next();
                        }
                        score += true_iter.next() == index_max;
                    }

                    return (score / y_true.size());
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> sparse_categorical_accuracy() noexcept {
            return std::shared_ptr<Metric<T>>(new SparseCategoricalAccuracy<T>());
        }

        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> sparse_categorical_accuracy(const std::string &name) {
            return std::shared_ptr<Metric<T>>(new SparseCategoricalAccuracy<T>(name));
        }
    }

    using namespace impl;
}



#endif