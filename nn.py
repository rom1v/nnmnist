#!/usr/bin/env python3
import matplotlib.pyplot as plt

import loader
import network
from network import Network

images_path = "data/train-images-idx3-ubyte.gz"
labels_path = "data/train-labels-idx1-ubyte.gz"
test_images_path = "data/t10k-images-idx3-ubyte.gz"
test_labels_path = "data/t10k-labels-idx1-ubyte.gz"


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def main():
    images = loader.load_mnist_images(images_path)
    labels = loader.load_mnist_labels(labels_path)
    test_images = loader.load_mnist_images(test_images_path)
    test_labels = loader.load_mnist_labels(test_labels_path)

    print(
        f"Loaded {images.shape[0]} images of shape {images.shape[1:]} and "
        f"{labels.shape[0]} labels."
    )
    print(
        f"Loaded {test_images.shape[0]} test images of shape {test_images.shape[1:]} "
        f"and {test_labels.shape[0]} labels."
    )

    training_data = Network.prepare_data(images, labels)
    test_data = Network.prepare_data(test_images, test_labels)

    nn = Network([28 * 28, 100, 10], cost=network.CrossEntropyCost)
    nn.train(
        training_data,
        mini_batch_size=10,
        eta=0.05,
        epochs=60,
        lmbda=5,
        test_data=test_data,
    )


if __name__ == "__main__":
    main()
