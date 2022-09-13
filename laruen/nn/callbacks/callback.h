
#ifndef LARUEN_NN_CALLBACKS_CALLBACK_H_
#define LARUEN_NN_CALLBACKS_CALLBACK_H_

#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"

namespace laruen::nn::callbacks {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::Model;

        template <typename T = float32_t>
        class Callback {
            private:
                Model<T> *model_;

            public:
                virtual ~Callback()
                {}

                virtual Callback<T>* clone() const = 0;

                virtual void operator()() const = 0;
        };
    }

    using namespace impl;
}

#endif