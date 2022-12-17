"""A module with preprocessing functions for the Histogram of Gradients (HOG) pipeline."""
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.transform import resize
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Flatten(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for flattening images."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, nx, ny = X.shape
        return X.reshape((n_samples, nx * ny))


class Normalizer(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for normalising an image if not already normalised."""

    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def fit(self, X):
        return self

    def transform(self, X):
        if not Normalizer.is_norm(X):
            X = (X - self.means) / self.stds

        return X

    @staticmethod
    def is_norm(img):
        return np.nansum(img) % 1 != 0


class ChannelSelector(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for selecting channels from an image."""

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X):
        return self

    def transform(self, X):
        return np.take(X, self.channels, axis=len(X.shape) - 1)


class NanReplacer(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for replacing nan values."""

    def fit(self, X):
        return self

    def transform(self, X):
        return np.nan_to_num(X, nan=0)


class DownSampler(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for downsampling."""

    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    def fit(self, X):
        return self

    def transform(self, X):
        downsampled = np.zeros((X.shape[0], self.new_width, self.new_height, X.shape[-1]))
        for i in range(X.shape[0]):
            downsampled[i] = downsample(X[i], self.new_width, self.new_height)
        return downsampled


class Gaussian(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for applying gaussian noise."""

    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X):
        return self

    def transform(self, X):
        for i in range(X.shape[0]):
            X[i] = apply_gaussian(X[i], self.sigma)
        return X


class Hog(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for creating a Histogram of Gradients."""

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        hogged = np.zeros(X.shape[:-1])
        for i in range(X.shape[0]):
            hogged[i] = create_hog(X[i])

        return hogged


class HogNpySaver(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for saving a Histogram of Gradients as .npy files."""

    def __init__(self, paths):
        self.paths = paths

    def fit(self, X):
        return self

    def transform(self, X):
        if len(self.paths) == 1:
            try:
                existing = np.load(*self.paths)
                to_save = np.vstack((existing, X))
                save_hog(to_save, *self.paths)
            except FileNotFoundError:
                save_hog(X, *self.paths)
            return
        for i, path in enumerate(self.paths):
            save_hog(X[i], path)


class HogCsvSaver(BaseEstimator, TransformerMixin):
    """A scikit pipeline step for saving a Histogram of Gradients into a single .csv file."""

    def __init__(self, name):
        self.name = name

    def fit(self, X):
        return self

    def transform(self, X):
        hog_rows = list(map(lambda arr: [arr], np.split(X, X.shape[0])))
        hogs = pd.DataFrame([*hog_rows], columns=['hog'])

        hogs.to_csv(f'{self.name}/hog.csv')


def downsample(image, new_width, new_height):
    """Returns a downsampled version of the input image with the new width and new height.

    :arg
    image (np.array) - a numpy array.
    new_width (int) - the new width of the image
    new_height (int) the new height of the image

    :return
    a numpy array of the downsampled image.
    """
    return resize(image, (new_width, new_height, image.shape[-1]))


def apply_gaussian(image, sigma=1):
    """Applies a gaussian filter to the image."""
    return gaussian(image, sigma=sigma)


def create_hog(image):
    """Creates a histogram of gradients from the provided image."""
    return \
        hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)[1]


def save_hog(hog_image, path):
    """Saves the hog image as a numpy array at the given path.

    :arg
    hog_image (np.array) - a numpy array.
    path - the path where the hog_image should be saved.
    """
    np.save(path, hog_image)
