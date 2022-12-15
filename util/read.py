"""A python module for reading data."""
import os
from glob import glob
import numpy as np
import logging


def get_img_file_names(start_dir):
    """Function for getting all .npy file names in a specific dir and all subdirs.

    :arg
    start_dir (str) - the directory in which the files will be looked for.

    :return
    a list of filenames.
    """
    files = []
    pattern = "*img_*.npy"

    for directory, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(directory, pattern)))

    return list(map(lambda filename: filename.replace('\\', '/'), files))


def load_arrays_batch(filenames, batch_size=1):
    """Function for loading numpy arrays from filenames.

    :arg
    filenames (iterable) - an iterable of filenames.
    lazy (bool) - True if a generator will be returned. False if a list with all arrays will be returned.
    batch_size (int) - for lazy loading, the batch_size indicates how many arrays will be returned at once.

    :return
    a tuple containing a generator of numpy arrays, and the filenames associated with them.
    """
    try:
        array_shape = (batch_size, *np.load(filenames[0]).shape)
        batch = np.zeros(array_shape)

        for i, filename in enumerate(filenames):
            batch[i % batch_size] = np.load(filename)

            if i + 1 == len(filenames):
                batch = np.delete(batch, (range(len(filenames) % batch_size, batch_size)), axis = 0)
                yield batch, filenames[i:i + batch_size]

            if (i + 1) % batch_size == 0:
                yield batch, filenames[i:i + batch_size]
                batch = np.zeros(array_shape)
    except ValueError as e:
        logging.warning(f'Problem while loading file: {filename}. {e}')


def load_arrays_all(filenames):
    """Function for loading numpy arrays from filenames.

    :arg
    filenames (iterable) - an iterable of filenames.

    :return
    a tuple containing a list of numpy arrays, and the filenames associated with them.
    """
    try:
        return np.array([np.load(filename) for filename in filenames]), filenames
    except ValueError as e:
        logging.warning(f'Problem while loading file: {filename}. {e}')
