"""A script for loading a mask file and saving it as a jpg."""
import argparse
import logging
import os
import sys

import numpy as np
import cv2

if __name__ == '__main__':
    # Setting up logging
    logging.root.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # Parsing the command line arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--mask", help="path of the mask to load")
    argParser.add_argument("-o", "--output", help="the output directory")

    args = argParser.parse_args()

    if not os.path.exists(args.mask):
        logging.error(f'The provided source path {args.mask} does not exist. Image cannot be loaded.')
        sys.exit(-1)

    image = np.load(args.mask)
    logging.info(f'Successfully loaded image from {args.mask}')

    # We want to add up the clean and debris glaciers
    image = np.sum(image[:, :, :2], axis=2)

    # create the output directory if it doesn't exist
    if not os.path.exists(args.output):
        logging.info(f"Output path {args.output} doesn't exist. Creating {args.output}")
        os.makedirs(args.output)

    image_output_name = 'mask'

    # Create the CV2 image
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)

    output_path = f'{args.output}/{args.mask.split("/")[-1].split(".")[0]}.jpg'
    cv2.imwrite(output_path, image * 255)
    logging.info(f'Image successfully saved at {output_path}')
