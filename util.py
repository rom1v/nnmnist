from scipy.fftpack import dct


def dct2d(image):
    return dct(dct(image.T, norm="ortho").T, norm="ortho")
