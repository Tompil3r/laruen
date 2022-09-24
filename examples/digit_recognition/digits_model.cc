
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

    // data: placeholders for inputs and labels
    // for training and testing the neural network.
    // data contains a total of 70,000 images and labels,
    // each image is 28 x 28 pixels (grayscale) and each label
    // is one single value. 60,000 images and labels are for
    // training and 10,000 for testing.
    // the data used in this example is the MNIST database
    // for handwritten digits. see: http://yann.lecun.com/exdb/mnist/
    // for source.
    multi::NDArray<> x_train(multi::Shape{60000, 28, 28});
    multi::NDArray<> x_test(multi::Shape{10000, 28, 28});
    multi::NDArray<> y_train(multi::Shape{60000, 1});
    multi::NDArray<> y_test(multi::Shape{10000, 1});

    // load datasets for memory to placeholders,
    // the datasets are saved as binary files, each pixel
    // value is extacly one byte.
    // the numbers 16 and 8 below are the offsets
    // for the actual data. bytes 0-15 and 0-7 in
    // the corresponding files contain information
    // about the structure of the data and is not needed.
    // the data files are not uploaded to this repository,
    // and in order to get them please download them from
    // the original source in the url listed above.
    x_train.load_buffer<uint8_t>("train_images_test", 16);
    x_test.load_buffer<uint8_t>("test_images_test", 16);
    y_train.load_buffer<uint8_t>("train_labels_test", 8);
    y_test.load_buffer<uint8_t>("test_labels_test", 8);

    // shuffle each dataset to remove any non-
    // random order which may affect training.
    nn::utils::batch_shuffle(x_train, y_train);
    nn::utils::batch_shuffle(x_test, y_test);

    // the digit recognition model contains 9 layers.
    // note that the first layer 'flatten' is included to
    // convert the 2d images into a 1d vector in order to make
    // the data the correct shape for the other layers.
    // different number of layers and different configurations can
    // be tested to get different results, (e.g: number of nodes in
    // each 'fully_connected' layer, or the hidden layers' activation function)
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

    // build the model according to a single input shape (without the batch dimension)
    // in this case, each input contains 1 image of a handwritten digits with the
    // dimensions 28 x 28.
    model.build({28, 28});

    // print the summary of the model - see the output at the end 
    model.summary();

    // compile the model with an 'adam' optimizer,
    // 'sparse_categorical_crossentropy' loss function and
    // 'sparse_cateogrical_accuracy' metric.
    // different optimizer configurations can be
    // tested for different results.
    model.compile(nn::optimizers::adam(),
    nn::losses::sparse_categorical_crossentropy("loss"),
    {nn::metrics::sparse_categorical_accuracy("accuracy")});

    // train the current model and retrieve it's
    // training 'History' object (currently unused)
    // with the following configurations:
    // - inputs to train on: x_train
    // - labels to train on: y_train
    // - callbacks: none
    // - batch size: 32
    // - epochs: 2
    // - validation split: 0.2 (take 0.2 of x_train & y_train
    // in order to validate the training by evaluating the model
    // on data that is not trained on)
    auto train_history = model.fit(x_train, y_train, {}, 32, 2, 0.2);

    // test the current model and retrieve it's evaluation
    // 'History' object (currently unused) with the following
    // configurations:
    // - inputs to evaluate on: x_test
    // - labels to evaluate on: y_test
    auto eval_history = model.evaluate(x_test, y_test);

    // print the prediction and labels for the first
    // 10 samples from the test dataset
    std::cout << model.predict(x_test[{{0, 10}}], 32, 0).indices_max({1}) << '\n' << std::endl;
    std::cout << y_test[{{0, 10}}] << std::endl;

    // save the weights of the model in the
    // specified file path (relative path)
    model.save_weights("examples/digit_recognition/digits_weights");

    return 0;

    // program compiled with:
    // g++ -O3 -g -I. -std=c++17 examples/digit_recognition/digits_model.cc -o digits_test

    /*
        program output:
        
        Layer Type        Output Shape        Params
        --------------------------------------------
        Flatten           (None, 784)         0
        Fully Connected   (None, 32)          25120
        ReLU              (None, 32)          0
        Fully Connected   (None, 64)          2112
        ReLU              (None, 64)          0
        Fully Connected   (None, 128)         8320
        ReLU              (None, 128)         0
        Fully Connected   (None, 10)          1290
        Softmax           (None, 10)          0

        Total Params: 36842
        --------------------------------------------
        epoch 1/2 - 375/375 - [====================] - loss: 1.0571 - accuracy: 0.7754 - val_loss: 0.5159 - val_accuracy: 0.8595
        epoch 2/2 - 375/375 - [====================] - loss: 0.3890 - accuracy: 0.8962 - val_loss: 0.3048 - val_accuracy: 0.9171
        epoch 1/1 - 313/313 - [====================] - loss: 0.3039 - accuracy: 0.9168
        [5, 2, 1, 0, 4, 1, 4, 9, 6, 9]

        [[5.000000]
        [2.000000]
        [1.000000]
        [0.000000]
        [4.000000]
        [1.000000]
        [4.000000]
        [9.000000]
        [5.000000]
        [9.000000]]
    */
}