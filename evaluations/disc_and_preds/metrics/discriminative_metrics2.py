

# Necessary Packages
import sys 
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, GRUCell, RNN, GRU, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, Mean
from tensorflow.keras.callbacks import Callback, EarlyStopping
# from utils import train_test_divide, extract_time, batch_generator


class Discriminator(Model):
    def __init__(self, seq_len, dim, hidden_dim, **kwargs): 
        super(Discriminator, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = dim
        self.hidden_dim = hidden_dim
        self.discriminator = self.build_discriminator()

        # self.model.compile(optimizer=Adam())
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.discriminator.compile(loss = self.cross_entropy, optimizer = self.optimizer)


    def build_discriminator(self,): 
        input_ = Input(shape=(self.seq_len, self.feat_dim), name='_input')
        x = input_

        self.gru = GRU(units = self.hidden_dim, activation = 'tanh', name='d_gru')
        self.flatten = Flatten(name='d_flatten')
        self.dense = Dense(units = 1)

        x = self.gru(x)
        x = self.flatten(x)
        output = self.dense(x)
        model = Model(input_, output, name = 'discriminator')
        return model


    def fit(self, X, y, epochs, verbose = 0, validation_split = None, shuffle = True, print_period=50):
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, min_delta = 1e-4, patience=100) 
        print_callback = PrintLossPerNthEpoch(label = 'Disc', print_period = print_period)

        iters = 2000
        batch_size = 128

        N = X.shape[0]
        epochs = iters * batch_size / N
        if epochs > 1: 
            steps_per_epoch = None
        else: 
            steps_per_epoch = iters

        epochs = math.ceil(iters * batch_size / N)

        Y_hat = self.discriminator.fit(X, y, 
            epochs = epochs, 
            verbose = verbose, 
            validation_split = validation_split,
            shuffle = shuffle,
            callbacks = [print_callback, early_stop_callback], 
            batch_size = batch_size,
            steps_per_epoch = steps_per_epoch
            )
        return Y_hat


    def __call__(self, X): 
        return self.predict(X)


    def predict(self, X):
        Y_hat = self.discriminator.predict(X)
        return Y_hat

    def summary(self):
        self.discriminator.summary()


def split_into_train_test(data, train_perc = 0.8): 
    N = data.shape[0]
    Ntrain = int(N * train_perc)
    indices = np.random.permutation(N)
    training_idx, test_idx = indices[:Ntrain], indices[Ntrain:]
    training, test = data[training_idx,:], data[test_idx,:]
    return training, test


class PrintLossPerNthEpoch(Callback):
    def __init__(self, label, print_period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.print_period = print_period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_period == (self.print_period - 1):
            try:  
                loss = np.round(logs['loss'], 3); val_loss = np.round(logs['val_loss'], 3)
                print( f"{self.label} Avg. train / val loss for epoch {epoch+1}: {loss} / {val_loss} " )
            except: 
                loss = np.round(logs['loss'], 3)
                print( f"{self.label} Avg. train loss for epoch {epoch+1}: {loss} " )
        else: 
            pass


def discriminative_score_metrics (ori_data, generated_data, epochs = 500, verbose = 0, print_epochs = 50):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data

    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    global print_period
    print_period = print_epochs

    # Basic Parameters
    N_real, seq_len, dim = ori_data.shape  
    N_gen = generated_data.shape[0]

    X = np.concatenate((ori_data, generated_data), axis=0)
    y = np.concatenate((np.ones([N_real,]), np.zeros([N_gen,])), axis = 0) 

    # shuffle both X and Y in unison
    X, y = shuffle (X, y)

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    ## Builde a post-hoc RNN discriminator network 
    # Network parameters
    hidden_dim = int(dim/2)    

    discriminator = Discriminator( 
            seq_len = seq_len, 
            dim = dim, 
            hidden_dim = hidden_dim
        )
    
    r = discriminator.fit(
        X_train, y_train, 
        epochs = epochs, 
        verbose = verbose,
        shuffle = True,
        validation_split=0.1,
        print_period = print_period
    )
    
    y_test_hat = discriminator.predict(X_test)

    # Compute the accuracy
    acc = accuracy_score(y_test, (y_test_hat>0.5))
    discriminative_score = np.abs(0.5-acc)
        
    return discriminative_score 

        


if __name__== '__main__': 

    real= np.random.randn(100, 24, 5)
    gen = np.random.randn(100, 24, 5)

    N, T, D = real.shape

    disc = Discriminator(
        seq_len = T, 
        dim = D, 
        hidden_dim = int(D/2)
    )

    disc.compile(optimizer = Adam(), loss='mse')

    disc.summary()

    outs = disc([real, gen])

    print(len(outs), outs[0].shape,outs[1].shape)

    disc.fit(
        real, gen,
        epochs =1000, 
        verbose = 1
    )

