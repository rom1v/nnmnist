import numpy as np
import random


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        diff = a - vectorize_output(y)
        return np.dot(diff.T, diff).squeeze()

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        y = vectorize_output(y)
        return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))

    @staticmethod
    def delta(z, a, y):
        return a - y


class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.sizes = sizes
        self.weights = [
            np.random.randn(rows, cols) for rows, cols in zip(sizes[1:], sizes[:-1])
        ]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.act_fn = sigmoid
        self.act_fn_prime = sigmoid_prime
        self.cost = cost

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.act_fn(w @ a + b)
        return a

    def train(
        self, training_data, mini_batch_size, eta, epochs, lmbda=0, test_data=None
    ):
        training_total = len(training_data)
        if test_data:
            test_total = len(test_data)
        for i in range(epochs):
            self.train_one_epoch(training_data, mini_batch_size, eta, lmbda)
            test_ok = self.evaluate(training_data)
            cost = self.compute_cost(training_data, lmbda)
            print(
                f"Epoch {i:2d} training: {test_ok:5d} / {training_total:5d} (cost={cost})"
            )
            if test_data:
                test_ok = self.evaluate(test_data)
                cost = self.compute_cost(test_data, lmbda)
                print(
                    f"Epoch {i:2d}     test: {test_ok:5d} / {test_total:5d} (cost={cost})"
                )
            else:
                print(f"Epoch {i:2d} complete")

    def train_one_epoch(self, training_data, mini_batch_size, eta, lmbda):
        n = len(training_data)
        random.shuffle(training_data)
        mini_batches = (
            training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
        )
        for mini_batch in mini_batches:
            self.train_mini_batch(mini_batch, eta, lmbda, n)

    def train_mini_batch(self, mini_batch, eta, lmbda, n):
        m = len(mini_batch)

        # x and y are matrices having 1 column for each input/output from the mini_batch
        x = np.concatenate([x for x, _ in mini_batch], axis=1)
        y = np.concatenate([vectorize_output(y) for _, y in mini_batch], axis=1)

        nabla_w, nabla_b = self.backprop(x, y)

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / m) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
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
        delta = self.cost.delta(z, a, y)

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

    def compute_cost(self, test_data, lmbda):
        cost = 0
        for x, y in test_data:
            a = self.feedforward(x)
            y_vec = vectorize_output(y)
            cost += self.cost.fn(a, y)
        # vector view, to compute the squared norm
        w_vecs = (w.reshape(-1) for w in self.weights)
        cost += lmbda / 2 * sum(np.dot(w, w) for w in w_vecs)
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


def vectorize_output(i):
    e = np.zeros((10, 1))
    e[i] = 1
    return e
