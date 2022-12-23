"""A script that generates a HOG from an image and saves it."""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog

from preprocessing.hog import ChannelSelector
from .channels import parse_channels

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

    channels = parse_channels(args.channels)

    if not os.path.exists(args.image):
        logging.error(f'The provided source path {args.image} does not exist. Image cannot be loaded.')
        sys.exit(-1)

    image = np.load(args.image)
    image = image.reshape((1, *image.shape))
    logging.info(f'Successfully loaded image from {args.image}')

    # create the output directory if it doesn't exist
    if not os.path.exists(args.output):
        logging.info(f"Output path {args.output} doesn't exist. Creating {args.output}")
        os.makedirs(args.output)

    for channel in channels:
        logging.info(f'Extracting channels: {channel} from image.')

        subset_image = ChannelSelector(channels=np.array(channel) - 1).fit_transform(image)
        hist, subset_image = hog(subset_image[0], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        subset_image *= 255
        subset_image = subset_image.astype(np.uint8)

        plt.hist(hist, color='teal')
        plt.ylabel('Gradient Magnitude')
        plt.xlabel('Orientation (Normalised)')

        image_output_name = '_'.join(map(str, channel))

        subset_image = Image.fromarray(subset_image)

        output_path = f'{args.output}/hog_{image_output_name}_{args.image.split("/")[-1].split(".")[0]}.jpg'
        plt.savefig(f'{args.output}/hist_{image_output_name}_{args.image.split("/")[-1].split(".")[0]}.eps', format='eps')
        subset_image.save(output_path)
        logging.info(f'Image successfully saved at {output_path}')
