"""Module for creating scikit pipelines."""
from sklearn.pipeline import Pipeline

from preprocessing.hog import NanReplacer, DownSampler, Gaussian, Hog, HogNpySaver, ChannelSelector, Normalizer


def saving_pipeline(means, stds, channels, width, height, sigma, path):
    saving = direct_pipeline(means, stds, channels, width, height, sigma)
    saving.steps.append(('saver', HogNpySaver([path])))

    return saving

def direct_pipeline(means, stds, channels, width, height, sigma):
    return Pipeline([
        ('normalise', Normalizer(means, stds)),
        ('channels', ChannelSelector(channels)),
        ('nan', NanReplacer()),
        ('sampler', DownSampler(width, height)),
        ('gaussian', Gaussian(sigma)),
        ('hog', Hog())
    ])