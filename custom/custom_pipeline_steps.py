from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from collections import defaultdict
import numpy as np
import pandas as pd

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