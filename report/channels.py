"""A script that reads an image and generates one image per channel subset"""
import argparse
import logging
import os
import sys

import cv2
import numpy as np

from preprocessing.hog import ChannelSelector

def parse_channels(channels):
    """Parses channel input into lists of channels.

    :arg
    channels (str) - a string of format channel,channel,channel:channel,....

    :return
    an iterable of parsed channels.
    """
    channels = channels.split(':')
    channels = map(lambda c: list(map(int, c.split(','))), channels)

    return channels


if __name__ == '__main__':
    # Setting up logging
    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # Parsing the command line arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--image", help="path of the image to load")
    argParser.add_argument("-c", "--channels", help="a list of lists of comma separated channels. Lists are separated with :. Example: 1,2,3:6,7,8:11,12,13")
    argParser.add_argument("-o", "--output", help="the output directory")

    args = argParser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        logging.error(f'The provided source path {image_path} does not exist. Image cannot be loaded.')
        sys.exit(-1)

    image = np.load(image_path)

    logging.info(f'Successfully loaded image from {image_path}')

    channels = parse_channels(args.channels)

    # create the output directory if it doesn't exist
    if not os.path.exists(args.output):
        logging.info(f"Output path {args.output} doesn't exist. Creating {args.output}")
        os.makedirs(args.output)

    for channel in channels:
        logging.info(f'Extracting channels: {channel} from image.')
        subset_image = ChannelSelector(channels=np.array(channel) - 1).fit_transform(image)

        for i in range(len(channel)):
            min_ = np.min(subset_image[:, :, i])
            max_ = np.max(subset_image[:, :, i])
            subset_image[:, :, i] = (subset_image[:, :, i] - min_) / (max_ - min_)

        image_output_name = '_'.join(map(str, channel))

        output_path = f'{args.output}/channels_{image_output_name}.jpg'
        cv2.imwrite(output_path, subset_image * 255)
        logging.info(f'Image successfully saved at {output_path}')
