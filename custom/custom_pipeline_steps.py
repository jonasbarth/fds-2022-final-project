from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from collections import defaultdict
import numpy as np
import pandas as pd
import os


class CustomNormalizer (BaseEstimator, TransformerMixin):
    def __init__ (self, means, std, path = 'og_dataset/splits/normalized/'):
        self.path = path
        self.means = means
        self.std = std
        
    def transform (self, X, y=None):
        X = X.copy()
        os.makedirs(self.path, exist_ok= True)
        for idx, row in X.iterrows():
            img = np.load(row.img_slice)
            if not CustomNormalizer.is_norm(img):
                img = (img - self.means)/self.std
                new_path = self.path + f'slice_{idx[0]}_img_{idx[1]}.npy'
                X.loc[idx,'img_slice'] = new_path
                np.save(new_path, img )
                
        return X
    
    def fit(self, X, y=None):
        return self
           
    @staticmethod 
    def is_norm (img):
        return np.nansum(img) % 1 != 0
        

class ColorHistogram(BaseEstimator,TransformerMixin):
    '''Pipeline step which computes the histogram of values for each of the layers in an image'''
    def __init__(self, binning='doane'):
        self.bins = dict()
        self.n_features = None
        
        # Set binning methods
        self.binning_param = binning
        self.binning = binning
            
    def fit(self, X, y=None):
        '''The expected shape of X is a dataset where each cell is an np array of values
            Columns are the channels
        '''
        self.n_features = X.shape[1]
        
        if type(self.binning) is not dict:
            # if only one binning method is specified, then it's used for all the layers
            self.binning = defaultdict(lambda: self.binning_param)

        for col_name in X.columns:
            # accumulate all values
            values = np.hstack(X[col_name])
            # remove NaN
            values = values[np.logical_not(np.isnan(values))]
            
            # compute the bins edges using the choosed binning method
            bins = np.histogram_bin_edges(values, bins=self.binning[col_name])
            self.bins[col_name] =  bins
        return self
    
    def transform(self, X, y=None):
        histos = dict()
        for col_name in X.columns:
            # select each column 
            C = np.stack(X[col_name])
            # apply the histogram function for each row
            hist = np.apply_along_axis(
                ColorHistogram.histogram,
                axis = 1,
                arr = C,
                bins = self.bins[col_name]
            )
            histos[col_name] = hist 
        return np.hstack(list(histos.values()))
    
    @staticmethod
    def histogram(slice, bins, density=True):
        slice = slice[np.logical_not(np.isnan(slice))]
        if len(slice) == 0:
            return np.full(len(bins)-1,np.nan)
        return np.histogram(slice, bins, density=density)[0]

    
def split(list_matrix, columns = []):
    '''Splits a 3D matrix into its layers'''
    df = list_matrix.apply(
        lambda m: [m[0][:,:,i].flat for i in range(m[0].shape[2])], 
        axis = 1,
        result_type='expand')
    if len(columns) == len(df.columns):
        df.columns = columns
    return df

Splitter = lambda cols: FunctionTransformer(func=split, kw_args= {'columns':cols})

def loader(file_list):
    '''Returns the loaded npy files'''
    return file_list.applymap(lambda path: np.load(path))

Loader = lambda: FunctionTransformer(func=loader)