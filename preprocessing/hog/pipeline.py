"""Module for creating scikit pipelines."""
from sklearn.pipeline import Pipeline

from preprocessing.hog import NanReplacer, DownSampler, Gaussian, Hog, HogNpySaver, ChannelSelector


def saving_pipeline(channels, width, height, sigma, path):
    saving = direct_pipeline(channels, width, height, sigma)
    saving.steps.append(('saver', HogNpySaver([path])))

    return saving

def direct_pipeline(channels, width, height, sigma):
    return Pipeline([
        ('channels', ChannelSelector(channels)),
        ('nan', NanReplacer()),
        ('sampler', DownSampler(width, height)),
        ('gaussian', Gaussian(sigma)),
        ('hog', Hog())
    ])