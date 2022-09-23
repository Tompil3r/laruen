
#ifndef LARUEN_NN_METRICS_BINARY_ACCURACY_H_
#define LARUEN_NN_METRICS_BINARY_ACCURACY_H_

#include <memory>
#include <string>
#include <type_traits>
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
        class BinaryAccuracy : public Metric<T> {
            private:
                T threshold_;
            
            public:
                template <typename = std::enable_if_t<std::is_arithmetic_v<T>>>
                inline BinaryAccuracy(T threshold = 0.5f)
                : Metric<T>("binary_accuracy"), threshold_(threshold)
                {}

                inline BinaryAccuracy(const std::string &name, T threshold = 0.5f)
                : Metric<T>(name), threshold_(threshold)
                {}

                inline Metric<T>* clone() const override final {
                    return new BinaryAccuracy<T>(*this);
                }

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);

                    T score = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        score += pred_iter.next() >= this->threshold_ ?
                        ((T)1.0f) == true_iter.next() : ((T)0.0f) == true_iter.next();
                    }

                    return (score / y_pred.size());
                }

                inline T optimizing_mode() const override final {
                    return 1.0;
                }
        };

        template <typename T = float32_t, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        inline std::shared_ptr<Metric<T>> binary_accuracy(T threshold = 0.5f) noexcept {
            return std::shared_ptr<Metric<T>>(new BinaryAccuracy<T>(threshold));
        }

        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> binary_accuracy(const std::string &name, T threshold = 0.5f) {
            return std::shared_ptr<Metric<T>>(new BinaryAccuracy<T>(threshold));
        }
    }

    using namespace impl;
}



#endif