import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [
            np.random.randn(rows, cols) for rows, cols in zip(sizes[1:], sizes[:-1])
        ]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.act_fn = sigmoid
        self.act_fn_prime = sigmoid_prime
        self.cost_fn = cross_entropy_cost
        self.cost_delta_fn = cross_entropy_cost_delta

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
                cost = self.cost(test_data)
                print(f"Epoch {i}: {test_ok} / {test_total} (cost={cost})")
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

        # x and y are matrices having 1 column for each input/output from the mini_batch
        x = np.concatenate([x for x, _ in mini_batch], axis=1)
        y = np.concatenate([vectorize_output(y) for _, y in mini_batch], axis=1)

        nabla_w, nabla_b = self.backprop(x, y)

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
        delta = self.cost_delta_fn(z, a, y)

        nabla_w[-1] = delta @ activations[-2].T
        # Sum all columns of delta into nabla_b[-1]
        nabla_b[-1] = delta @ np.ones((delta.shape[1], 1))

        num_layers = len(self.weights) + 1
        for l in range(2, num_layers):
            z = zs[-l]
            delta = (self.weights[-l + 1].T @ delta) * self.act_fn_prime(z)
            nabla_w[-l] = delta @ activations[-l - 1].T
            # Sum all columns of delta into nabla_b[-l]
            nabla_b[-l] = delta @ np.ones((delta.shape[1], 1))

        return (nabla_w, nabla_b)

    def evaluate(self, test_data):
        test_results = ((np.argmax(self.feedforward(x)), y) for x, y in test_data)
        return sum(int(x == y) for x, y in test_results)

    def cost(self, test_data):
        cost = 0
        for x, y in test_data:
            a = self.feedforward(x)
            y_vec = vectorize_output(y)
            cost += self.cost_fn(a, y)
        cost /= len(test_data)
        return cost

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
        return [(images[i].reshape(pixels, 1), labels[i]) for i in range(n)]


def sigmoid(z):
    z = np.clip(z, -100, 100)  # avoid exp overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def quadratic_cost(a, y):
    diff = a - vectorize_output(y)
    return np.dot(diff.T, diff).squeeze()


def quadratic_cost_delta(z, a, y):
    return (a - y) * sigmoid_prime(z)


def cross_entropy_cost(a, y):
    y = vectorize_output(y)
    return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))


def cross_entropy_cost_delta(z, a, y):
    return a - y


def vectorize_output(i):
    e = np.zeros((10, 1))
    e[i] = 1
    return e
