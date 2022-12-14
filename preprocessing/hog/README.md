# Preprocessing - Histogram of Gradients

## Purpose
This package contains modules for preprocessing glacier image data. We want to turn the glacier image data into 
histogram of gradients.

## Structure
The package contains the following modules:
* `hog`: contains functions and classes to be used in a scikit pipeline, for image downsampling, applying gaussian filters, replacing NaN values, and saving data.
* `pipeline`: contains functions for create scikit pipelines. Useful if you want to have a single pipeline for preprocessing and model fitting.
* `main`: a script that preprocesses the data.

## Preprocessing Script
Our image data is > 10GB and is difficult to fit into RAM. For this reason, we created a script that does all the 
necessary preprocessing in batches and saves the data with its labels, such that it is ready to use for a classifier. 
After preprocessing, the data footprint is reduced by a factor of 50.

### Prerequisites

#### 1. Data
You need to have the image data on your machine.

#### 2. Configuration
The script relies on a config file to tell it where to find the data, save the histograms, and what parameters to use 
during the preprocessing. You need to make sure that the paths specified in the `config.yaml` file point to the folders 
of your data. Fill in the rest of the parameters to your liking.
   
Example:
```yaml
source:
  path: # the path where to data will be loaded from.
    dev: data/image/dev
    test: data/image/test
    train: data/image/train
hog:
  path: # the path where the data will be saved.
    dev: data/hog/dev
    test: data/hog/test
    train: data/hog/train
  normalising: dataset/channel_mean_std.csv
  channels: # the different channels that will be used. In the end, there will be one dataset per channel.
    - [1, 2, 3]
    - [6, 7, 8]
    - [11, 12, 13]
```

### How to run
The script is a python script that takes two command line arguments:
- `-c` for the path to the config file.
- `-m` for the path to the metadata file.

The script needs these two files to be able to run properly. Running it from the root of the project for example:
```python
python -m preprocessing.hog.main -c preprocessing/hog/config.yaml -m dataset/metadata.csv
```
