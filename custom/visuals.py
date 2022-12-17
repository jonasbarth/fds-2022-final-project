import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .env_variables import *
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
    '''Shows an img and its mask side by side'''
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
    
def show_hist(hist, kind='RGB', **params):
    '''Shows a band selection of an histogram of colors'''
    gh = hist.reset_index(level=0, name='density')
    gh.rename(columns={'level_0':'channel'}, inplace=True)
    gh = gh[(gh.density!=0) & (gh.channel.isin(bands_combination[kind]))]
    g = sns.barplot(data=gh, y='density', x=gh.index, hue='channel', palette=bands_palette[kind], **params)
    sns.despine(bottom=True)
    g.set(xticklabels=[])
    g.tick_params(bottom=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title('Histogram of Colors')
    plt.xlabel(kind+' channels')
    return g

def show_img_hist(img, hist, channel='RGB'):
    '''Shows an image and its histogram side by side'''
    plt.figure(figsize=(10,20))
    plt.subplot(1,2,1)
    show(img, kind=channel)
    ax2 = plt.subplot(1,2,2)
    show_hist(hist, kind=channel)
    # keep same size for both subplots
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    plt.tight_layout()
    plt.show()

