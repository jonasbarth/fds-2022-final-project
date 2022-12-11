"""A python script for finding images that contain nan values."""
import numpy as np
import pandas as pd

from util.read import load_arrays, get_img_file_names

if __name__ == '__main__':
    filenames = get_img_file_names('../data')

    any_nan_files = []
    all_nan_files = []

    for patch, filename in load_arrays(filenames):
        filename = filename.replace("\\", "/").split("/")[-1]
        isnan = np.isnan(patch)
        if isnan.all():
            all_nan_files.append(filename)
        elif isnan.any():
            any_nan_files.append(filename)

    pd.DataFrame(any_nan_files, columns=['filename']).to_csv('any_nan_files.csv')
    pd.DataFrame(all_nan_files, columns=['filename']).to_csv('all_nan_files.csv')