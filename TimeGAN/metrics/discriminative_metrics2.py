# Necessary Packages
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import sys
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, GRUCell, RNN, GRU, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, Mean
from tensorflow.keras.callbacks import Callback, EarlyStopping


class Discriminator():
    def __init__(self, seq_len, dim, hidden_dim):
        self.seq_len = seq_len
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.discriminator = self.build_model()        

        # self.model.compile(optimizer=Adam())
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.discriminator.compile(loss = self.cross_entropy, optimizer = self.optimizer, run_eagerly=True)

    
    def build_model(self): 
        input_ =  Input(shape=(self.seq_len, self.dim), name='disc_input')        
        x = GRU(units = self.hidden_dim, activation = 'tanh', name='d_gru')(input_)
        x = Flatten(name='d_flatten')(x)
        x = Dense(units = 1)(x)        
        output = tf.squeeze(x)
        model = Model(input_, output, name = 'discriminator')
        return model
      

    def fit(self, X, y, epochs, verbose = 0, validation_split = None, shuffle = True):
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, min_delta = 1e-4, patience=100) 
        print_callback = PrintLossPerNthEpoch(print_period = 50)

        Y_hat = self.discriminator.fit(X, y, 
            epochs = epochs, 
            verbose = verbose, 
            validation_split = validation_split,
            shuffle = shuffle,
            callbacks = [print_callback]
            )
        return Y_hat


    def __call__(self, X): 
        return self.predict(X)


    def predict(self, X): 
        Y_hat = self.discriminator.predict(X, batch_size=X.shape[0])
        return Y_hat

    def summary(self):
        self.discriminator.summary()


class PrintLossPerNthEpoch(Callback):
    def __init__(self, print_period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_period = print_period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_period == 0:
            try:  
                loss = np.round(logs['loss'], 3); val_loss = np.round(logs['val_loss'], 3)
                print( f"Avg. train / val loss for epoch {epoch}: {loss} / {val_loss} " )
            except: 
                loss = np.round(logs['loss'], 3)
                print( f"Avg. train loss for epoch {epoch}: {loss} " )
        else: 
            pass


def discriminative_score_metrics (ori_data, generated_data, epochs = 500, verbose = 0):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data

    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Basic Parameters
    N_real, seq_len, dim = ori_data.shape  
    N_gen = generated_data.shape[0]

    X = np.concatenate((ori_data, generated_data), axis=0)
    y = np.concatenate((np.ones([N_real,]), np.zeros([N_gen,])), axis = 0) 

    # shuffle both X and Y in unison
    X, y = shuffle (X, y)
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    ## Builde a post-hoc RNN discriminator network 
    # Network parameters
    hidden_dim = int(dim/2)    
    batch_size = 128

    discriminator = Discriminator(seq_len = seq_len, 
        dim = dim, 
        hidden_dim = int(dim/2)
        )
    
    r = discriminator.fit(
        X_train, y_train, 
        epochs = epochs, 
        verbose = verbose,
        shuffle = True,
        validation_split=0.1
    )

    y_test_hat = discriminator.predict(X_test)

    # Compute the accuracy
    acc = accuracy_score(y_test, (y_test_hat>0.5))
    discriminative_score = np.abs(0.5-acc)
        
    return discriminative_score 

        


if __name__== '__main__': 

    real= np.random.randn(100, 24, 5)
    gen = np.random.randn(100, 24, 5)
    N_real, T, D = real.shape
    N_gen = gen.shape[0]

    X = np.concatenate((real, gen), axis=0)
    Y = np.concatenate((np.ones([N_real,]), np.zeros([N_gen,])), axis = 0)
    # print('orig shapes', X.shape, Y.shape); sys.exit()

    disc = Discriminator(
        seq_len = T, 
        dim = D, 
        hidden_dim = int(D/2)
    )
    disc.summary()

    y_hat = disc.predict(X)
    # print(y_hat.shape); sys.exit()

    disc.fit(
        X, Y,
        epochs =1000, 
        verbose = 0
    )

