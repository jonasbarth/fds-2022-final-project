# Final Project
Final Project for Fundamentals of Data Science 2022.

# Contributors
* Nemish Murawat
* Matteo Migliarino
* Mattia Castaldo
* Javier Martinez
* Jonas Barth

# Data
https://drive.google.com/drive/folders/1l-0gi2-_5sYK6fYMycisbETqrSNeA0Xs?usp=share_link

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
