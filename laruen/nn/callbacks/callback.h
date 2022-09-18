
#ifndef LARUEN_NN_CALLBACKS_CALLBACK_H_
#define LARUEN_NN_CALLBACKS_CALLBACK_H_

#include <cstdint>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"

namespace laruen::nn {

    namespace impl {
        template <typename> class Model;
    }

    namespace callbacks {

        namespace impl {

            using laruen::multi::NDArray;
            using laruen::multi::float32_t;
            using laruen::nn::Model;

            template <typename T = float32_t>
            class Callback {
                protected:
                    Model<T> *model_ = nullptr;
                    uint_fast8_t verbose_mode_;
                    std::string verbose_;

                public:
                    virtual ~Callback()
                    {}

                    Callback(uint_fast8_t verbose = 1)
                    : verbose_mode_(verbose)
                    {}

                    inline void model(Model<T> *model) noexcept {
                        this->model_ = model;
                    }

                    inline Model<T>* model() noexcept {
                        return this->model_;
                    }

                    inline void verbose_mode(uint_fast8_t verbose) {
                        this->verbose_mode_ = verbose;
                    }

                    inline uint_fast8_t verbose_mode() const {
                        return this->verbose_mode_;
                    }

                    inline std::string& verbose() noexcept {
                        return this->verbose_;
                    }

                    inline virtual void set(Model<T> *model) {
                        this->model_ = model;
                    }

                    virtual Callback<T>* clone() const = 0;

                    virtual void on_epoch_end(uint_fast64_t epoch)
                    {};
            };
        }

        using namespace impl;
    }
}

#endif