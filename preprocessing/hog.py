"""A module with preprocessing functions for the Histogram of Gradients (HOG) pipeline."""
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.transform import resize


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
    return hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)[1]
# TODO function for saving histograms
