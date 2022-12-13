"""Module for creating scikit pipelines."""
from sklearn.pipeline import Pipeline

from preprocessing.hog import NanReplacer, DownSampler, Gaussian, Hog, HogNpySaver

def saving_pipeline(width, height, sigma, path):
    saving = direct_pipeline(width, height, sigma)
    saving.steps.append(('saver', HogNpySaver([path])))

    return saving

def direct_pipeline(width, height, sigma):
    return Pipeline([
        ('nan', NanReplacer()),
        ('sampler', DownSampler(width, height)),
        ('gaussian', Gaussian(sigma)),
        ('hog', Hog())
    ])