"""This is a script that converts the stats_train.json file into a csv.

Example:
    python script/channel_mean_std.py -f ./data/stats_train.json -t ./script_test.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--from_path", help="path of the file to read from")
    argParser.add_argument("-t", "--to_path", help="path where to save the converted csv file")

    args = argParser.parse_args()

    print(f'Reading file from {args.from_path}')

    with open(args.from_path, 'r') as f:
        data = json.load(f)
        means = np.array(data['means'])
        stds = np.array(data['stds'])

    channels = np.array(range(1, 16))

    data = {'channel': channels, 'means': means, 'stds': stds}

    print(f'Writing file to {args.to_path}')
    pd.DataFrame(data).to_csv(args.to_path)