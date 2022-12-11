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

    return files


def load_arrays(filenames, lazy=True):
    """Function for loading numpy arrays from filenames.

    :arg
    filenames (iterable) - an iterable of filenames.
    lazy (bool) - True if a generator will be returned. False if a list with all arrays will be returned.

    :return
    a tuple containing a list or generator of numpy arrays, and the filenames associated with them.
    """

    try:
        if lazy:
            for filename in filenames:
                yield np.load(filename), filename

        else:
            return [(np.load(filename), filename) for filename in filenames]
    except ValueError as e:
        logging.warning(f'Problem while loading file: {filename}. {e}')
