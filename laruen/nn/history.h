
#ifndef LARUEN_NN_HISTORY_H_
#define LARUEN_NN_HISTORY_H_

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include "laruen/nn/metrics/metric.h"

namespace laruen::nn {

    namespace impl {

        using laruen::nn::metrics::Metric;

        template <typename T>
        class History {
            std::map<std::string, std::vector<T>> metrics_map_;

            public:
                History(const std::vector<std::shared_ptr<Metric<T>>> &metrics)
                {
                    for(auto metric = metrics.cbegin();metric != metrics.cend();metric++) {
                        this->metrics_map_.insert({(*metric)->name(), (*metric)->values()});
                    }
                }

                std::string str() const noexcept {
                    std::string string("History {\n");

                    for(const auto& [metric, values]: this->metrics_map_) {

                        string.append(4, ' ');
                        string.append(metric);
                        string.append(": [");

                        auto value = values.cbegin();

                        for(;value != values.cend() - 1;value++) {
                            string.append(std::to_string(*value));
                            string.push_back(',');
                            string.push_back(' ');
                        }

                        string.append(std::to_string(*value));
                        string.append("]\n");
                    }

                    string.push_back('}');

                    return string;
                }

                friend inline std::ostream& operator<<(std::ostream &stream, const History &history) noexcept {
                    return stream << history.str();
                }
        };
    }

    using namespace impl;
}


#endif