import numpy as np
import gzip
import struct


def load_mnist_images(file_path):
    """Load MNIST image from a file_path

    Return a ndarray containing a list of 2D images. The array shape is (count, rows, cols),
    for example (60000, 28, 28). Each value is a uint8 representing the pixel intensity.
    """
    with gzip.open(file_path, "rb") as f:
        headers = f.read(16)
        magic, count, rows, cols = struct.unpack(">IIII", headers)

        if magic != 0x803:
            raise ValueError(f"Invalid magic number {magic:x}, expected 0x803")

        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8)
        # Devide by 255 to use pixels values between 0 and 1 (otherwise learning fails)
        images = images.reshape((count, rows, cols)) / 255

        return images


def load_mnist_labels(file_path):
    """Load MNIST labels from a file_path

    Return a numpy array contaning the list of labels (uint8). The array shape is (count),
    for example (60000).
    """
    with gzip.open(file_path, "rb") as f:
        headers = f.read(8)
        magic, count = struct.unpack(">II", headers)

        if magic != 0x801:
            raise ValueError(f"Invalid magic number {magic:x}, expected 0x801")

        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)

        return labels
