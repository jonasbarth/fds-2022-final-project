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

        for images, names in tqdm(load_arrays_batch(filenames, batch_size=10)):
            hog_paths = list(map(lambda filename: f'{hog_path}/{filename.split("/")[-1]}', names))

            p = saving_pipeline(f'{hog_path}/hog.npy')

            transformed = p.fit_transform(images)


        labels = []

        for filename in filenames:
            filename = filename.split('/')[-1]

            label = metadata[metadata.img_slice == filename].label.values[0]
            labels.append(label)

        labels = np.array([labels]).T
        np.save(f'{hog_path}/labels.npy', labels)
