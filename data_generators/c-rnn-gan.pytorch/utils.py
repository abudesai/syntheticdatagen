
import numpy as np, pandas as pd
import matplotlib.pyplot as plt 
import sys


TITLE_FONT_SIZE = 16

def get_data(dataset_type, training_size, data_dir): 
    fname = f'{data_dir + dataset_type}_subsampled_train_perc_{training_size}.npz'
    loaded = np.load(fname)
    data = loaded['data']
    return data


def get_train_valid_split(data, valid_perc):
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train

    # shuffle data, just in case
    np.random.shuffle(data)

    # train, valid split 
    train_data = data[:N_train]
    valid_data = data[N_train:]
    return train_data, valid_data


def scale_train_valid_data(train_data, valid_data): 
    
    _, T, D = train_data.shape
    
    scaler = MinMaxScaler()        
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    
    return scaled_train_data, scaled_valid_data, scaler


def plot_samples(samples, n, title):    
    fig, axs = plt.subplots(n, 1, figsize=(6,8))
    idxes = []
    for i in range(n):
        rnd_idx = np.random.choice(len(samples)); idxes.append(rnd_idx)
        s = samples[rnd_idx]
        axs[i].plot(s)    
    print(idxes)
    fig.suptitle(title, fontsize = TITLE_FONT_SIZE)
    # fig.tight_layout()
    plt.show()


class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        self.mini = np.expand_dims(self.mini, 0)
        self.range = np.expand_dims(self.range, 0)
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-10)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data



#########################################################################


class DataLoader:
    def __init__(self, dataset_type, training_size, data_dir,  
            batch_size, valid_perc, test_perc, do_shuffle=True):
        fname = f'{data_dir + dataset_type}_subsampled_train_perc_{training_size}.npz'
        loaded = np.load(fname)
        data = loaded['data']
        data = data.astype(dtype=np.float32)

        # data = data[:, :, :1]

        self.batch_size = batch_size
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
        self.songs['validation'] = data[train_len: train_len + valid_len]
        self.songs['test'] = data[train_len + valid_len: ]

        self.scaler = MinMaxScaler( )  
        self.songs['train'] = self.scaler.fit_transform(self.songs['train'])
        if valid_len > 0:  self.songs['validation'] = self.scaler.transform(self.songs['validation'])
        if test_len > 0: self.songs['test'] = self.scaler.transform(self.songs['test'])


    def get_rescaled_data(self, data): 
        return self.scaler.inverse_transform(data)

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
        if self.pointer[part] == 0 and len(self.songs[part]) - batchsize < 0: 
            self.pointer[part] = len(self.songs[part])
            return [None, self.songs[part]]

        if self.pointer[part] > len(self.songs[part]) - batchsize:
            return [None, None]

        batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
        self.pointer[part] += batchsize

        if songlength > self.T: 
            raise Exception(f"Error sampling series. Target length {songlength} exceeds available length {self.T}")
        elif songlength == self.T: 
            sampled_data = batch
        else:
            sampled_data = []
            for i in range(batch.shape[0]):
                rand_idx = np.random.randint(0, self.T - songlength)
                sampled_data.append( np.expand_dims( batch[i, rand_idx: rand_idx + songlength, :] , axis=0 ) )        
            sampled_data = np.concatenate(sampled_data, axis=0)

        return [None, sampled_data]

    def get_orig_num_samples(self): 
        return self.N


    def get_batch2(self): 
        return next(self.batch_gen)


    def get_num_seq_features(self):
        return self.D

    def get_seq_len(self): 
        return self.T

    def get_num_meta_features(self):
        return None


    def save_data(self, filename, samples):
        # samples_fpath = f'{model}_gen_samples_{data_name}_perc_{p}.npz'        
        # np.savez_compressed(os.path.join( output_dir, samples_fpath), data=samples)
        pass




class DataLoader2:
    def __init__(self, dataset_type, training_size, data_dir,  
            batch_size, valid_perc, test_perc, used_len, do_shuffle=True):
        fname = f'{data_dir + dataset_type}_subsampled_train_perc_{training_size}.npz'
        loaded = np.load(fname)
        data = loaded['data']
        data = data.astype(dtype=np.float32)

        data = data[:, :used_len, :]

        # data = data[:, :, :1]

        self.batch_size = batch_size
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
        self.songs['validation'] = data[train_len: train_len + valid_len]
        self.songs['test'] = data[train_len + valid_len: ]

        self.scaler = MinMaxScaler( )  
        self.songs['train'] = self.scaler.fit_transform(self.songs['train'])
        if valid_len > 0:  self.songs['validation'] = self.scaler.transform(self.songs['validation'])
        if test_len > 0: self.songs['test'] = self.scaler.transform(self.songs['test'])


    def get_rescaled_data(self, data): 
        return self.scaler.inverse_transform(data)

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
        if self.pointer[part] == 0 and len(self.songs[part]) - batchsize < 0: 
            self.pointer[part] = len(self.songs[part])
            return [None, self.songs[part]]

        if self.pointer[part] > len(self.songs[part]) - batchsize:
            return [None, None]

        batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
        self.pointer[part] += batchsize

        if songlength > self.T: 
            raise Exception(f"Error sampling series. Target length {songlength} exceeds available length {self.T}")
        elif songlength == self.T: 
            sampled_data = batch
        else:
            sampled_data = []
            for i in range(batch.shape[0]):
                rand_idx = np.random.randint(0, self.T - songlength)
                sampled_data.append( np.expand_dims( batch[i, rand_idx: rand_idx + songlength, :] , axis=0 ) )        
            sampled_data = np.concatenate(sampled_data, axis=0)

        return [None, sampled_data]

    def get_orig_num_samples(self): 
        return self.N


    def get_batch2(self): 
        return next(self.batch_gen)


    def get_num_seq_features(self):
        return self.D

    def get_seq_len(self): 
        return self.T

    def get_num_meta_features(self):
        return None


    def save_data(self, filename, samples):
        # samples_fpath = f'{model}_gen_samples_{data_name}_perc_{p}.npz'        
        # np.savez_compressed(os.path.join( output_dir, samples_fpath), data=samples)
        pass

    