
#ifndef LARUEN_NN_CALLBACKS_WEIGHTS_SAVING_H_
#define LARUEN_NN_CALLBACKS_WEIGHTS_SAVING_H_

#include <cstdint>
#include <string>
#include "laruen/nn/callbacks/callback.h"
#include "laruen/nn/metrics/metric.h"

namespace laruen::nn::callbacks {

    namespace impl {

        using laruen::nn::metrics::Metric;
        
        template <typename T>
        class WeightsSaving : public Callback<T> {
            private:
                const Metric<T> *monitor_;
                std::string filepath_;
                bool save_best_only_;
                uint_fast64_t save_freq_;
                std::size_t filepath_format_pos_;
                T mode_;
                T best_monitored_ = -1.0;
                
            public:
                WeightsSaving(const Metric<T> *monitor, const std::string &filepath, T mode,
                bool save_best_only = false, uint_fast64_t save_freq = 1, uint_fast8_t verbose = 1)
                : Callback<T>(verbose), monitor_(monitor), filepath_(filepath),
                save_best_only_(save_best_only), save_freq_(save_freq),
                filepath_format_pos_(filepath.rfind("{}")), mode_(mode)
                {}

                inline WeightsSaving(const Metric<T> &monitor, const std::string &filepath,
                bool save_best_only = false, uint_fast64_t save_freq = 1, uint_fast8_t verbose = 1)
                : WeightsSaving<T>(&monitor, filepath, monitor.optimizing_mode(), save_best_only, save_freq, verbose)
                {}

                inline Callback<T>* clone() const override final {
                    return new WeightsSaving<T>(*this);
                }

                bool on_epoch_end(uint_fast64_t epoch) override final {
                    auto formated_filepath = [](std::string &filepath,
                    std::size_t format_pos, uint_fast64_t epoch) -> std::string
                    {
                        std::string formated_path(filepath.cbegin(), filepath.cbegin() + format_pos);
                        formated_path.append(std::to_string(epoch));
                        formated_path.append(filepath.cbegin() + format_pos + 2, filepath.cend());

                        return formated_path;
                    };

                    if(this->save_best_only_) {
                        T curr_metric_value = this->monitor_->values()[epoch];

                        // on first epoch or when improvement occurs
                        if(!epoch || ((curr_metric_value - this->best_monitored_) * this->mode_ > 0)) {
                            this->best_monitored_ = curr_metric_value;

                            if(this->filepath_format_pos_ != std::string::npos) {
                                this->model_->save_weights(
                                    formated_filepath(this->filepath_, this->filepath_format_pos_, epoch));
                            }
                            else {
                                this->model_->save_weights(this->filepath_);
                            }
                        }
                    }
                    else if(!(epoch % this->save_freq_)) {
                        if(this->filepath_format_pos_ != std::string::npos) {
                            this->model_->save_weights(
                                formated_filepath(this->filepath_, this->filepath_format_pos_, epoch));
                        }
                        else {
                            this->model_->save_weights(this->filepath_);
                        }
                    }

                    return false;
                }

                inline void monitor(const Metric<T> *monitor) noexcept {
                    this->monitor_ = monitor;
                }

                inline const Metric<T>* monitor() const noexcept {
                    return this->monitor_;
                }

                inline void filepath(const std::string &filepath) noexcept {
                    this->filepath_ = filepath;
                }

                inline const std::string& filepath() const noexcept {
                    return this->filepath_;
                }

                inline void mode(T mode) {
                    this->mode_ = mode;
                }

                inline T mode() const noexcept {
                    return this->mode_;
                }

                inline void save_best_only(bool save_best_only) noexcept {
                    this->save_best_only_ = save_best_only;
                }

                inline bool save_best_only() const noexcept {
                    return this->save_best_only_;
                }

                inline void save_freq(uint_fast64_t save_freq) noexcept {
                    this->save_freq_ = save_freq;
                }

                inline uint_fast64_t save_freq() const noexcept {
                    return this->save_freq_;
                }

                inline std::size_t filepath_format_pos() const noexcept {
                    return this->filepath_format_pos_;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Callback<T>> weights_saving(const Metric<T> &monitor,
        const std::string &filepath, bool save_best_only = false,
        uint_fast64_t save_freq = 1, uint_fast8_t verbose = 1)
        {
            return std::shared_ptr<Callback<T>>(new WeightsSaving<T>(monitor,
            filepath, save_best_only, save_freq, verbose));
        }
    }

    using namespace impl;
}


#endif