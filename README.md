# Final Project
Final Project for Fundamentals of Data Science 2022.

# Contributors
* Nemish Murawat
* Matteo Migliarini
* Mattia Castaldo
* Javier Martinez
* Jonas Barth

# Resources
[Paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2020/57/paper.pdf) that we base the project on.

# Data (ask Jonas to start lab to make the data available)
AWS S3 Links:
* [dev](https://fds-final-project.s3.amazonaws.com/dev.zip)
* [test](https://fds-final-project.s3.amazonaws.com/test.zip)
* [train](https://fds-final-project.s3.amazonaws.com/train.zip)

# Final Report
Editing link to the final report on overleaf: https://www.overleaf.com/2312131596hnsgrggxmgpp

# Installing Requirements

## Conda
They should all be installed already.

## pip
```
pip install -r requirements.txt
```

# Structure

![project_structure](doc/structure.svg)

## Data
These are our image patches from the 35 Landsat satellite images.

## Classifiers
Our goal is to *determine whether an image contains a glacier or not*. For this we have multiple classifiers.
Each of the classifiers will need receive differently preprocessed data. For example, the _KNN_ will "train" on a
a [Histogram of Oriented Gradients](https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f),
whereas the _CNN_ might train on the actual image with a selection of channels.

### 1. KNN

#### Preprocess

#### Fit/Train

#### Evaluate

### 2. CNN

#### Preprocess

#### Fit/Train

#### Evaluate

### 3. SVM (Maybe)

#### Preprocess

#### Fit/Train

#### Evaluate
