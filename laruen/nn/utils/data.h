#ifndef LARUEN_NN_UTILS_DATA_H_
#define LARUEN_NN_UTILS_DATA_H_

#include <tuple>
#include <cstdint>
#include <type_traits>
#include <random>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/rng.h"
#include "laruen/nn/utils/utils.h"

namespace laruen::nn::utils {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T>
        inline void data_split(NDArray<T> &data, NDArray<T> &batch_data,
        uint_fast64_t size, uint_fast64_t offset = 0)
        {
            batch_data = batch_view(data, size);
            batch_data.data(batch_data.data() + offset);
        }

        template <typename T>
        inline void train_test_val_split(NDArray<T> &data, NDArray<T> &train, NDArray<T> &test,
        NDArray<T> &val, uint_fast64_t train_size, uint_fast64_t test_size, uint_fast32_t val_size = 0)
        {
            data_split(data, train, train_size, 0);
            data_split(data, test, test_size, train_size * data.strides().front());
            data_split(data, val, val_size, (train_size + val_size) * data.strides().front());
        }

        template <typename T>
        inline void train_test_val_split(NDArray<T> &data, NDArray<T> &train, NDArray<T> &test,
        NDArray<T> &val, float32_t train_portion, float32_t test_portion, float32_t val_portion = 0.0)
        {
            uint_fast64_t batch_size = data.shape().front();

            train_test_val_split(data, train, test, val, (uint_fast64_t)(batch_size * train_portion),
            (uint_fast64_t)(batch_size * test_portion), (uint_fast64_t)(batch_size * val_portion));
        }

        template <typename T, typename TT, typename =
        std::enable_if_t<!(std::is_same_v<TT, uint_fast64_t> || std::is_same_v<TT, float32_t>)>>
        inline void train_test_val_split(NDArray<T> &data, NDArray<T> &train, NDArray<T> &test,
        NDArray<T> &val, TT train_size, TT test_size, TT val_size = 0)
        {
            using size_type = std::conditional_t<std::is_integral_v<TT>, uint_fast64_t, float32_t>;

            return train_test_val_split(data, train, test, val,
            (size_type)train_size, (size_type)test_size, (size_type)val_size);
        }


        template <typename T>
        inline std::tuple<NDArray<T>, NDArray<T>, NDArray<T>> train_test_val_split(NDArray<T> &data,
        uint_fast64_t train_size, uint_fast64_t test_size, uint_fast32_t val_size = 0)
        {
            std::tuple<NDArray<T>, NDArray<T>, NDArray<T>> arrays;

            train_test_val_split(data, std::get<0>(arrays), std::get<1>(arrays),
            std::get<2>(arrays), train_size, test_size, val_size);

            return arrays;
        }
        
        template <typename T>
        inline std::tuple<NDArray<T>, NDArray<T>, NDArray<T>> train_test_val_split(NDArray<T> &data,
        float32_t train_portion, float32_t test_portion, float32_t val_portion = 0.0)
        {
            uint_fast64_t batch_size = data.shape().front();

            return train_test_val_split(data, (uint_fast64_t)(batch_size * train_portion),
            (uint_fast64_t)(batch_size * test_portion), (uint_fast64_t)(batch_size * val_portion));
        }

        template <typename T, typename TT, typename =
        std::enable_if_t<!(std::is_same_v<TT, uint_fast64_t> || std::is_same_v<TT, float32_t>)>>
        inline std::tuple<NDArray<T>, NDArray<T>, NDArray<T>> train_test_val_split(NDArray<T> &data,
        TT train_size, TT test_size, TT val_size = 0)
        {
            using size_type = std::conditional_t<std::is_integral_v<TT>, uint_fast64_t, float32_t>;

            return train_test_val_split(data, (size_type)train_size, (size_type)test_size, (size_type)(val_size));
        }

        template <typename T>
        inline void batch_shuffle(NDArray<T> &x, NDArray<T> &y,
        std::mt19937::result_type seed = std::random_device{}())
        {
            laruen::multi::RNG.seed(seed);
            x.shuffle(0);
            laruen::multi::RNG.seed(seed);
            y.shuffle(0);
        }

    }

    using namespace impl;
}






#endif