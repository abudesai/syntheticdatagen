import os, random, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from datetime import timedelta
import math
import tensorflow as tf


class MissingTimeIntervalFiller():
    ''' Adds missing time intervals in a time-series dataframe.     '''
    DAYS = 'days'
    MINUTES = 'minutes'
    HOURS = 'hours'

    def __init__(self, id_columns, time_column, value_col_prefix, time_unit, step_size, fill_na_val ):
        super().__init__()
        if not isinstance(id_columns, list):
            self.id_columns = [id_columns]
        else:
            self.id_columns = id_columns

        self.time_column = time_column

        self.val_col_prefix = value_col_prefix

        self.time_unit = time_unit
        self.step_size = int(step_size)
        self.fill_na_val = fill_na_val

    
    def fit(self, X, y=None): return self # do nothing in fit
        

    def transform(self, X):
        
        value_columns = [ c for c in X.columns if c.startswith(self.val_col_prefix) ]

        min_time = X[self.time_column].min()
        max_time = X[self.time_column].max() 

        if self.time_unit == MissingTimeIntervalFiller.DAYS:
            num_steps = ( (max_time - min_time).days // self.step_size ) + 1
            all_time_ints = [min_time + timedelta(days=x*self.step_size) for x in range(num_steps)]

        elif self.time_unit == MissingTimeIntervalFiller.HOURS:
            time_diff_sec = (max_time - min_time).total_seconds()
            num_steps =  int(time_diff_sec // (3600 * self.step_size)) + 1
            num_steps = (max_time - min_time).days + 1
            all_time_ints = [min_time + timedelta(hours=x*self.step_size) for x in range(num_steps)]

        elif self.time_unit == MissingTimeIntervalFiller.MINUTES:
            time_diff_sec = (max_time - min_time).total_seconds()
            num_steps =  int(time_diff_sec // (60 * self.step_size)) + 1
            # print('num_steps', num_steps)
            all_time_ints = [min_time + timedelta(minutes=x*self.step_size) for x in range(num_steps)]
        else: 
            raise Exception(f"Unrecognized time unit: {self.time_unit}. Must be one of ['days', 'hours', 'minutes'].")

        # create df of all time intervals
        full_intervals_df = pd.DataFrame(data = all_time_ints, columns = [self.time_column])  

        # get unique id-var values from original input data
        id_cols_df = X[self.id_columns].drop_duplicates()
        
        # get cross join of all time intervals and ids columns
        full_df = id_cols_df.assign(foo=1).merge(full_intervals_df.assign(foo=1)).drop('foo', 1)

        # merge original data on to this full table
        full_df = full_df.merge(X[self.id_columns + [self.time_column] + value_columns], on=self.id_columns + [self.time_column], how='left')

        # fill na values
        if self.fill_na_val is not None: 
            full_df[value_columns] = full_df[value_columns].fillna(self.fill_na_val) 

        return full_df

    
# Custom scaler for 3d data
class TSMinMaxScaler():
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, input_dim, upper_bound = 3., lower_bound = -3.):         
        self.scaling_len = scaling_len
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        

    def fit(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)

        self.min_vals_per_d = np.expand_dims( X[ :,  : self.scaling_len , : ].min(axis=1), axis = 1)
        self.max_vals_per_d = np.expand_dims( X[ :,  : self.scaling_len , : ].max(axis=1), axis = 1)
        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d

        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)
              
        return self
    
    def transform(self, X, y=None): 
        assert X.shape[0] == self.min_vals_per_d.shape[0], "Error: Dimension of array to scale doesn't match fitted array."
        assert X.shape[2] == self.min_vals_per_d.shape[2], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X[:, :, :self.input_dim] = np.where( X[:, :, :self.input_dim] < self.upper_bound, X[:, :, :self.input_dim], self.upper_bound)

        X[:, :, :self.input_dim] = np.where( X[:, :, :self.input_dim] > self.lower_bound, X[:, :, :self.input_dim], self.lower_bound)
        return X
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X.copy()
        X[:, :, : self.input_dim] = X[:, :, : self.input_dim ] * self.range_per_d[:, :, : self.input_dim] 
        X[:, :, : self.input_dim] = X[:, :, : self.input_dim] + self.min_vals_per_d[:, :, : self.input_dim]
        # print(X.shape)
        return X


# Custom scaler for 3d data
class MinMaxScaler_Feat_Dim():
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, input_dim, upper_bound = 3., lower_bound = -3.):         
        self.scaling_len = scaling_len
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        

    def fit(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)

        X_f = X[ :,  : self.scaling_len , : ]
        self.min_vals_per_d = np.expand_dims(np.expand_dims(X_f.min(axis=0).min(axis=0), axis=0), axis=0)
        self.max_vals_per_d = np.expand_dims(np.expand_dims(X_f.max(axis=0).max(axis=0), axis=0), axis=0)

        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d
        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)

        # print(self.min_vals_per_d.shape); print(self.max_vals_per_d.shape)
              
        return self
    
    def transform(self, X, y=None): 
        assert X.shape[-1] == self.min_vals_per_d.shape[-1], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X = np.where( X < self.upper_bound, X, self.upper_bound)
        X = np.where( X > self.lower_bound, X, self.lower_bound)
        return X
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X.copy()
        X = X * self.range_per_d 
        X = X + self.min_vals_per_d
        # print(X.shape)
        return X

    
class DailyAggregator():
    ''' Aggregates time-series values to daily level.     '''
    def __init__(self, id_columns, time_column, value_col_prefix, exog_col_prefix ):
        super().__init__()
        if not isinstance(id_columns, list):
            self.id_columns = [id_columns]
        else:
            self.id_columns = id_columns

        self.time_column = time_column

        self.val_col_prefix = value_col_prefix
        self.exog_col_prefix = exog_col_prefix


    def fit(self, X, y=None): return self


    def transform(self, X):
        X = X.copy()
        X[self.time_column] = X[self.time_column].dt.normalize()

        value_columns = [ c for c in X.columns if c.startswith(self.val_col_prefix) ]
        exog_cols = [ c for c in X.columns if c.startswith(self.exog_col_prefix) ]
        groupby_cols = self.id_columns + [self.time_column]
        sum_cols = value_columns + exog_cols
        X = X.groupby(by=groupby_cols, as_index=False)[sum_cols].sum()        
        return X
    
    
# to generate sines data
def generate_sine_data(no, seq_len, dim):   
    """Sine data generation.
    Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data
    """ 
    size = (no, 1, dim)
    freq = np.random.uniform(0, 0.1, size)   
    phase = np.random.uniform(-1, 1, size)
    
    seq = np.arange(seq_len)
    seq = np.expand_dims(seq, axis=0)
    seq = np.expand_dims(seq, axis=-1)
    
    data = np.sin(freq * seq + phase)
    
    data = (data + 1) * 0.5
    return data


def scale_data(data):
    scaler = TSMinMaxScaler(
        forecast_len = forecast_length,
        input_dim = input_dim,
        upper_bound = scaler_upper_bound
    )
    scaled_data = min_max_scaler_sine.fit_transform(data)
#     print(scaled_data.shape, scaled_data.mean(axis=0).mean(axis=0))
    return scaled_data, scaler