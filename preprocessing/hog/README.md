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
You need to make sure that the paths specified in the `config.yaml` file point to the folders of your data.
```yaml
source:
  path:
    dev: data/image/dev
    test: data/image/test
    train: data/image/train
hog:
  path:
    dev: data/hog/dev
    test: data/hog/test
    train: data/hog/train
  channels:
    - [1, 2, 3]
    - [6, 7, 8]
    - [11, 12, 13]
```

### How to run
The script is a simple python script. Run it from the `hog` package.
```python
python main.py
```