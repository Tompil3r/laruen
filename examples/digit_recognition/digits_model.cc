
#include <iostream>
#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/nn/model.h"
#include "laruen/nn/layers/fully_connected.h"
#include "laruen/nn/layers/softmax.h"
#include "laruen/nn/layers/relu.h"
#include "laruen/nn/layers/flatten.h"
#include "laruen/nn/losses/sparse_categorical_crossentropy.h"
#include "laruen/nn/optimizers/adam.h"
#include "laruen/nn/metrics/sparse_categorical_accuracy.h"
#include "laruen/nn/utils/data.h"

int main() {
    using namespace laruen;

    multi::NDArray<> x_train(multi::Shape{60000, 28, 28});
    multi::NDArray<> x_test(multi::Shape{10000, 28, 28});
    multi::NDArray<> y_train(multi::Shape{60000, 1});
    multi::NDArray<> y_test(multi::Shape{10000, 1});

    x_train.load_buffer<uint8_t>("train_images_test", 16);
    x_test.load_buffer<uint8_t>("test_images_test", 16);
    y_train.load_buffer<uint8_t>("train_labels_test", 8);
    y_test.load_buffer<uint8_t>("test_labels_test", 8);

    nn::utils::batch_shuffle(x_train, y_train);
    nn::utils::batch_shuffle(x_test, y_test);

    nn::Model model({
        nn::layers::flatten(),
        nn::layers::fully_connected(32),
        nn::layers::relu(),
        nn::layers::fully_connected(64),
        nn::layers::relu(),
        nn::layers::fully_connected(128),
        nn::layers::relu(),
        nn::layers::fully_connected(10),
        nn::layers::softmax()
    });

    model.build({28, 28});

    model.summary();

    model.compile(nn::optimizers::adam(),
    nn::losses::sparse_categorical_crossentropy("loss"),
    {nn::metrics::sparse_categorical_accuracy("accuracy")});

    auto train_history = model.fit(x_train, y_train, {}, 32, 2, 0.2);

    auto eval_history = model.evaluate(x_test, y_test);

    std::cout << model.predict(x_test[{{0, 10}}], 32, 0).indices_max({1}) << '\n' << std::endl;
    std::cout << y_test[{{0, 10}}] << std::endl;

    model.save_weights("examples/digit_recognition/digits_weights");

    return 0;
}