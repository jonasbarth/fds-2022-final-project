import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .env_variables import feature_names, bands_combination, channels_stats
from .custom_pipeline_steps import CustomNormalizer

def minmax_img(img):
    mins = np.array(channels_stats['mins'])
    maxs = np.array(channels_stats['maxs'])
    return (img - mins)/(maxs - mins)

def denormalize(img):
    return img * channels_stats['stds'] + channels_stats['means']

def load_img(file, unnormalize=True, minmax=True):
    '''Loads a numpy slice into memory and unnormalize it if requested'''
    img = np.load(file)
    if unnormalize and CustomNormalizer.is_norm(img):
        img = denormalize(img)
    if minmax:
        img = minmax_img(img)
    return img

def show(img, kind='RGB'):
    '''Shows an image'''
    if type(kind) is str:
        channels = [feature_names.index(chan) for chan in bands_combination[kind]]
    else:
        channels = kind
    img = img[:,:,channels]
    plt.axis('off')
    return plt.imshow(img)

def show_img_mask(img, mask):
    plt.subplot(1,2,1)
    show(img, kind='RGB') 
    plt.title('Image RGB')
    plt.subplot(1,2,2)
    show(mask, kind=[0])
    plt.title('Mask')
    plt.show()


def show_bands(img, mask):
    '''Shows all the bands we are interested in for a given image'''
    bands_num = len(bands_combination) + 1
    plt.subplot(1, bands_num, 1)
    plt.title('Glaciers:')
    show(mask, kind=0)
    for i, (name, band) in enumerate(bands_combination.items()):
        plt.subplot(1, bands_num, i+2)
        plt.title(name)
        show(img, kind=name)
    plt.show()
    
def show_hist(img, hist, channel):
    plt.subplot(1,2,1)
    show(img)
    plt.subplot(1,2,2)
    hist = hist.loc[feature_names[channel]]
    plt.bar(x=hist.index, height=hist)
    plt.show()