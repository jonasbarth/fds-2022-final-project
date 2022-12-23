"""A script for preprocessing image data for a classifier that takes Histogram of Gradients."""
import argparse
import logging
import os
import sys

import munch
import numpy as np
import pandas as pd
import yaml

from preprocessing.hog import saving_pipeline
from util import load_arrays_batch, get_img_file_names

if __name__ == '__main__':
    # Setting up the logger
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # Parsing the command line arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", help="path of the config to load")
    argParser.add_argument("-m", "--metadata", help="path of the metadata file to load")

    args = argParser.parse_args()

    # Reading config files
    config = munch.munchify(yaml.safe_load(open(args.config)))

    metadata = pd.read_csv(args.metadata)
    metadata.img_slice = metadata.img_slice.apply(lambda path: path.split('/')[-1])

    normalising = pd.read_csv(config.hog.normalising)
    means = normalising.means.to_numpy()
    stds = normalising.stds.to_numpy()

    # the metadata that we will save at the end of the preprocessing
    hog_metadata = {'channels': [], 'type': [], 'data_path': [], 'label_path': []}

    # go through the dev, test, train, folders.
    for source_path, hog_path_key in zip(vars(config.source.path).values(), vars(config.hog.path)):

        if not os.path.exists(source_path):
            logging.error(f'The provided source path {source_path} does not exist. Image data cannot be loaded.')
            sys.exit(-1)

        hog_metadata['type'] += [hog_path_key] * len(config.hog.channels)
        hog_path = config.hog.path[hog_path_key]

        logging.info(f"Reading image files from {source_path}")
        filenames = get_img_file_names(source_path)

        # create the output directory if it doesn't exist
        if not os.path.exists(hog_path):
            logging.info(f"Output path {hog_path} doesn't exist. Creating {hog_path}")
            os.makedirs(hog_path)

        # go through each combination of channels
        for channel in config.hog.channels:

            hog_metadata['channels'].append(channel)

            # subtract 1 since the channels in the config are 1-indexed.
            channel = np.array(channel)
            channel = channel - 1

            file_name_suffix = '_'.join(map(str, channel))
            data_output_path = f'{hog_path}/hog_{file_name_suffix}.npy'
            hog_metadata['data_path'].append(data_output_path)

            for images, names in load_arrays_batch(filenames, batch_size=10):
                p = saving_pipeline(means, stds, channel, 214, 214, 1, data_output_path)

                p.fit_transform(images)

            logging.info(f"Saved HOG to: {data_output_path}.")

        # create a labels file for each image folder (dev, test, train).
        labels = []

        logging.info(f'Creating labels for {hog_path}.')
        for filename in filenames:
            filename = filename.split('/')[-1]

            label = metadata[metadata.img_slice == filename].label.values[0]
            labels.append(label)

        labels = np.array(labels)
        labels_output_path = f'{hog_path}/labels.npy'
        np.save(labels_output_path, labels)
        logging.info(f'Saved labels to: {labels_output_path}.')

        hog_metadata['label_path'] += [labels_output_path] * len(config.hog.channels)

        # Create and save a dataframe containing HOG metadata
        pd.DataFrame(hog_metadata).to_csv(f'{config.hog.meta}/metadata.csv')
