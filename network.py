import numpy as np
import random
import util

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [
            np.random.randn(rows, cols) for rows, cols in zip(sizes[1:], sizes[:-1])
        ]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.act_fn = sigmoid
        self.act_fn_prime = sigmoid_prime
        self.cost_fn_prime = cost_prime

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.act_fn(w @ a + b)
        return a

    def train(self, training_data, mini_batch_size, eta, epochs, test_data=None):
        if test_data:
            test_total = len(test_data)
        for i in range(epochs):
            self.train_one_epoch(training_data, mini_batch_size, eta)
            if test_data:
                test_ok = self.evaluate(test_data)
                print(f"Epoch {i}: {test_ok} / {test_total}")
            else:
                print(f"Epoch {i} complete")

    def train_one_epoch(self, training_data, mini_batch_size, eta):
        n = len(training_data)
        random.shuffle(training_data)
        mini_batches = (
            training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
        )
        for mini_batch in mini_batches:
            self.train_mini_batch(mini_batch, eta)

    def train_mini_batch(self, mini_batch, eta):
        m = len(mini_batch)
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        a = x  # current activation
        activations = [a]
        zs = []  # all z vectors
        # same as feedforward, but store the intermediate z
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            zs.append(z)
            a = self.act_fn(z)
            activations.append(a)

        # a and z contains the values for the last layer
        delta = self.cost_fn_prime(a, y) * self.act_fn_prime(z)

        nabla_w[-1] = delta @ activations[-2].T
        nabla_b[-1] = delta

        num_layers = len(self.weights) + 1
        for l in range(2, num_layers):
            z = zs[-l]
            delta = (self.weights[-l + 1].T @ delta) * self.act_fn_prime(z)
            nabla_w[-l] = delta @ activations[-l - 1].T
            nabla_b[-l] = delta
        return (nabla_w, nabla_b)

    def evaluate(self, test_data):
        test_results = ((np.argmax(self.feedforward(x)), y) for x, y in test_data)
        return sum(int(x == y) for x, y in test_results)

    @staticmethod
    def prepare_data(images, labels):
        """Reformat data to be used by this network.

        Input:
         - images is an ndarray having shape (count, rows, cols)
         - labels is an ndarray having shape (count)

        Output:
         - python list of tuples (image, label), where:
           - image is an ndarray (rows*cols, 1)
           - label is a number
        """
        n = len(images)
        pixels = images.shape[1] * images.shape[2]
        return [(util.dct2d(images[i]).reshape(pixels, 1), labels[i]) for i in range(n)]


def sigmoid(z):
    z = np.clip(z, -100, 100)  # avoid exp overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def cost_prime(output, y):
    return output - vectorize(y)


def vectorize(i):
    e = np.zeros((10, 1))
    e[i] = 1
    return e
