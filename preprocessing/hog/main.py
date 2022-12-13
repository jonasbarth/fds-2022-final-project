"""A script for preprocessing image data for a classifier that takes Histogram of Gradients."""

import munch
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing.hog import saving_pipeline
from util import load_arrays_batch, get_img_file_names

if __name__ == '__main__':

    config = munch.munchify(yaml.safe_load(open("config.yaml")))

    metadata = pd.read_csv('./dataset/metadata.csv')
    metadata.img_slice = metadata.img_slice.apply(lambda path: path.split('/')[-1])

    for source_path, hog_path in zip(vars(config.preprocess.source.path).values(), vars(config.preprocess.hog.path).values()):
        filenames = get_img_file_names(source_path)
        # TODO create hog data for each channel
        for channel in config.preprocess.hog.channels:
            # subtract 1 since the channels in the config are 1-indexed.
            channel = np.array(channel)
            file_name_suffix = '_'.join(map(str, channel))
            channel = channel - 1
            for images, names in tqdm(load_arrays_batch(filenames, batch_size=10)):

                p = saving_pipeline(channel, 214, 214, 1, f'{hog_path}/hog_{file_name_suffix}.npy')

                transformed = p.fit_transform(images)


        labels = []

        for filename in filenames:
            filename = filename.split('/')[-1]

            label = metadata[metadata.img_slice == filename].label.values[0]
            labels.append(label)

        labels = np.array([labels]).T
        np.save(f'{hog_path}/labels.npy', labels)
