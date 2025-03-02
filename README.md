# Neutral Networks 101

This is an implementation of the classical example of recognizing handwritten
digits from the MNIST dataset.

This is an exercice from [Neural Networks and Deep Learning][nndl], written by
[Michael Nielsen].

The algorithm is also explained in the first 4 videos of the [Neural
Networks][3b1b] playlist by _3blue1brown_.

[nndl]: http://neuralnetworksanddeeplearning.com
[Michael Nielsen]: https://michaelnielsen.org/

[3b1b]: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi


## Run

Just execute:

```bash
./nn.py
```

For simplicity, the datasets are provided in `data/`.


## Expected results

By default, it uses one hidden layer with 30 neurons, and trains the network on
the first 50,000 images of the MNIST training dataset. The network is then
evaluated on the 10,000 test images.

The results should look like this:


```console
$ ./nn.py
Loaded 60000 images of shape (28, 28) and 60000 labels.
Loaded 10000 test images of shape (28, 28) and 10000 labels.
Epoch 0: 9068 / 10000
Epoch 1: 9226 / 10000
Epoch 2: 9317 / 10000
Epoch 3: 9330 / 10000
Epoch 4: 9371 / 10000
Epoch 5: 9419 / 10000
Epoch 6: 9435 / 10000
Epoch 7: 9477 / 10000
Epoch 8: 9463 / 10000
Epoch 9: 9423 / 10000
Epoch 10: 9500 / 10000
Epoch 11: 9476 / 10000
Epoch 12: 9469 / 10000
Epoch 13: 9489 / 10000
Epoch 14: 9491 / 10000
Epoch 15: 9470 / 10000
Epoch 16: 9488 / 10000
Epoch 17: 9491 / 10000
Epoch 18: 9514 / 10000
Epoch 19: 9484 / 10000
Epoch 20: 9484 / 10000
Epoch 21: 9502 / 10000
Epoch 22: 9505 / 10000
Epoch 23: 9494 / 10000
Epoch 24: 9494 / 10000
Epoch 25: 9508 / 10000
Epoch 26: 9524 / 10000
Epoch 27: 9500 / 10000
Epoch 28: 9508 / 10000
Epoch 29: 9490 / 10000
```

It correctly classifies about 98.2% of the test images.


## Deep convolutional network

Chapter 6 of the book presents a convolutional network.

In this repository, it is implemented using [PyTorch] in `./nn_convnet.py` (the
book used _Theano_).

[PyTorch]: https://pytorch.org/

It correctly classifies about 99% of the test images:

```
Test Error:
 Accuracy: 99.0%, Avg loss: 0.035133
```
