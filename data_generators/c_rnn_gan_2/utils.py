import os, random, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf



def get_sine_data_samples(no, seq_len, dim):   
    """Sine data generation.
    Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data
    """ 
    np.random.seed(1)
    size = (no, 1, dim)
    freq = np.random.uniform(0, 0.1, size)    
    phase = np.random.uniform(-1., 1., size)
    
    seq = np.arange(seq_len)
    seq = np.expand_dims(seq, axis=0)
    seq = np.expand_dims(seq, axis=-1)
    
    data = np.sin(freq * seq + phase)
    
    data = (data + 1) * 0.5
    return data


def save_gen_data(data):
    cols = [f'c{i}' for i in range(data.shape[1])]
    data_df = pd.DataFrame(data, columns=cols)
    data_df.to_csv('./outputs/gen_data.csv', index=False)


def get_data(data_path):
    data = np.load(data_path)
    return data


#########################################################################


class DataLoader:
    def __init__(self, data_path, batch_size, valid_perc, test_perc, do_shuffle=True):
        self.batch_size = batch_size
        data = np.load(data_path)
        data = data.astype(dtype=np.float32)
        self.valid_perc = valid_perc
        self.test_perc = test_perc

        if do_shuffle:  np.random.shuffle(data)

        self.num_batches = data.shape[0] // batch_size

        self.N, self.T, self.D = data.shape
        self.data = data
        self.batch_gen = self.create_batch_gen()
        self.pointer = { 'train': 0, 'validation': 0, 'test': 0, }     
        
        valid_len = test_len = 0        
        if valid_perc:
            valid_len = int(float(self.valid_perc/100.0)* self.N)
        if test_perc:
            test_len = int(float(test_perc/100.0)* self.N)

        train_len = self.N - valid_len - test_len
        self.songs = {}
        self.songs['train'] = data[0: train_len]
        self.songs['valid'] = data[train_len: train_len + valid_len]
        self.songs['test'] = data[train_len + valid_len: ]

    def create_batch_gen(self):
        def gen():
            while True: # run through one round of batches            
                N2 = self.num_batches * self.batch_size
                idx = np.random.choice(np.arange(self.N), size=N2, replace=False)
                sampled_data = self.data[idx, :]
                for i in range(self.num_batches):
                    batch_data = sampled_data[i * self.batch_size: (i+1) * self.batch_size, : ]
                    yield batch_data
        return gen()

    
    def rewind(self, part='train'):
        self.pointer[part] = 0

    
    def get_batch(self, batchsize, songlength, part='train'):
        if self.pointer[part] > len(self.songs[part]) - batchsize:
            return [None, None]

        batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
        self.pointer[part] += batchsize

        if songlength > self.T: 
            raise Exception(f"Error sampling series. Target length {songlength} exceeds available length {self.T}")
        else:
            sampled_data = []
            for i in range(batch.shape[0]):
                rand_idx = np.random.randint(0, self.T - songlength)
                sampled_data.append( np.expand_dims( batch[i, rand_idx: rand_idx + songlength, :] , axis=0 ) )        
            sampled_data = np.concatenate(sampled_data, axis=0)

        return [None, sampled_data]


    def get_batch2(self): 
        return next(self.batch_gen)


    def get_num_seq_features(self):
        return self.D


    def get_num_meta_features(self):
        return None



class MinMaxScaler():
    '''Scales time-series based on history data 
     Scaling_len defines history length'''
    def __init__(self, scaling_len, upper_bound = 5.):         
        self.scaling_len = scaling_len
        self.upper_bound = upper_bound
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.range_per_d = None  
        

    def fit(self, X, y=None): 
        curr_len = X.shape[1]

        if curr_len < self.scaling_len: 
            msg = f''' Error scaling series. 
            scaling_len ({self.scaling_len}) needs to be <= array_len ({curr_len}). '''
            raise Exception(msg)

        self.min_vals_per_d = np.expand_dims( X[ :,  : self.scaling_len , : ].min(axis=1), axis = 1)
        self.max_vals_per_d = np.expand_dims( X[ :,  : self.scaling_len , : ].max(axis=1), axis = 1)
        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d

        self.range_per_d = np.where(self.range_per_d == 0, 1e-9, self.range_per_d)              
        return self

    
    def transform(self, X, y=None):         
        assert X.shape[0] == self.min_vals_per_d.shape[0], "Error: Dimension of array to scale doesn't match fitted array."
        assert X.shape[2] == self.min_vals_per_d.shape[2], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X = np.where( X < self.upper_bound, X, self.upper_bound)
        return X
    

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X * self.range_per_d
        X = X + self.min_vals_per_d
        return X



class SeriesSampler():
    ''' Samples a sub-series of length T' <= the original series of length T. 
    expects an array of shape NxTxD.

    Returns array of shape (N', T', D)
    where N' is requested number of samples, and T' is the requested sub series length
    '''
    def __init__(self, sample_len, num_samples_per_series): 
        self.sample_len = sample_len
        self.num_samples_per_series = num_samples_per_series


    def fit(self, X, y=None): return self


    def transform(self, X):
        curr_len = X.shape[1]

        if curr_len <= self.sample_len: 
            msg = f''' Error sampling  series. 
            sample_len ({self.sample_len}) needs to be <= given array_len ({curr_len}). '''
            raise Exception(msg)
        else:
            sampled_data = []
            for _ in range(self.num_samples_per_series):
                for i in range(X.shape[0]):
                    rand_idx = np.random.randint(0, curr_len - self.sample_len)
                    sampled_data.append( np.expand_dims( X[i, rand_idx: rand_idx + self.sample_len, :] , axis=0 ) )        
            sampled_data = np.concatenate(sampled_data, axis=0)        
        return sampled_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


#########################################################################

def get_dataset(data_arr, batch_size):
    X = data_arr.copy().astype(dtype=np.float32)
    np.random.shuffle(X)

    num_batches = X.shape[0] // batch_size
    X = X[: num_batches*batch_size]

    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    return dataset


#########################################################################

def process_energy_data():
    file = './data/energy_data.csv'
    data = pd.read_csv(file).values
    print('orig', data.shape)

    data = np.reshape(data, newshape=(1, -1, data.shape[-1]))
    print('reshaped', data.shape, data.mean())

    scaler = MinMaxScaler(scaling_len=data.shape[1])
    data = scaler.fit_transform(data)
    print('scaled', data.shape, data.min())

    sampler = SeriesSampler(sample_len = 24, num_samples_per_series=1000)
    data = sampler.fit_transform(data)
    print('sampled', data.shape, data.min())

    np.save('./data/energy_data_npy.npy', data)
    return 



def process_stock_data():    
    file = './data/stock_data.csv'
    data = pd.read_csv(file).values
    print('orig', data.shape)
    

    data = np.reshape(data, newshape=(1, -1, data.shape[-1]))
    print('reshaped', data.shape, data.mean())

    scaler = MinMaxScaler(scaling_len=data.shape[1])
    data = scaler.fit_transform(data)
    print('scaled', data.shape, data.min())

    sampler = SeriesSampler(sample_len = 24, num_samples_per_series=1000)
    data = sampler.fit_transform(data)
    print('sampled', data.shape, data.min())

    np.save('./data/stock_data_npy.npy', data)
    return 


#########################################################################



def review_gen_sine_data(): 
    file = './outputs/sine_data/generated_data_epoch_81.npy'

    data = np.load(file)

    plot_samples(data)
    
    if data.shape[-1] == 1:
        data = np.squeeze(data)
        print(data.shape)
        pd.DataFrame(data).to_csv("./outputs/gen_data_squeezed.csv", index=False)


def review_gen_sine_data(): 
    file = './outputs/sine_data/generated_data_epoch_81.npy'

    data = np.load(file)

    plot_samples(data)
    
    if data.shape[-1] == 1:
        data = np.squeeze(data)
        print(data.shape)
        pd.DataFrame(data).to_csv("./outputs/gen_data_squeezed.csv", index=False)


def review_gen_data(dir_path, file):
    file = f'{dir_path}/{file}'
    print(file)

    data = np.load(file)
    plot_samples(data, save_path=dir_path)

    data = data[:, :, 0]
    pd.DataFrame(data).to_csv("./outputs/gen_data_squeezed.csv", index=False)


def plot_samples(data, save_path, max_num_series=5, max_len = 100 ):
    
    N = min(data.shape[0], max_num_series)
    T = min(data.shape[1], max_len)
    D = 0

    min_val, max_val = data.min(), data.max()
    min_val = math.floor(min_val *100 ) / 100
    max_val = math.ceil(max_val *100 ) / 100
    # print(min_val, max_val)

    idx = np.random.randint(data.shape[0], size=N)
    sampled_data = data[idx,:, D]

    fig, axs = plt.subplots(nrows=N, ncols=1, figsize = (14, 2*N), sharex=True, sharey=True)
    for i in range(N):
        data_arr = sampled_data[i]

        if sampled_data.shape[1] - T  > 0:
            rand_idx = np.random.randint(0, sampled_data.shape[1] - T)
            data_arr = data_arr[rand_idx: rand_idx + T]

        ax = axs[i] if N > 1 else axs[0]
        ax.plot(data_arr)  
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_ylim([min_val, max_val])

    plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=.4, wspace=0.4)
    plt.show()
    # fig.tight_layout()
    fig.savefig(f'{save_path}/sample_data.png')


#########################################################################

if __name__ == '__main__':
    # data_path = './data/sine_data_arr.npy'
    # data = get_data(data_path)
    # print(data.shape)

    # # dataset = get_dataset(data, batch_size=32)
    # # print(dataset)

    # loader = DataLoader(data_path=data_path, batch_size=32, valid_perc=10, test_perc=10)

    

    # for i in range(20): 
    #     batch = loader.get_batch()
    #     print(i, batch.shape, batch.sum())


    # review_gen_data()


    # process_energy_data()
    # process_stock_data()


    dir_path = './outputs/energy_data'
    file = 'generated_data_epoch_285.npy'
    review_gen_data(dir_path, file)