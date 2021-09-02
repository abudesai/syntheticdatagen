
# Necessary Packages
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, GRUCell, RNN, GRU, Flatten, Dense, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, Mean
from tensorflow.keras.callbacks import Callback, EarlyStopping
import sys 
# from utils import train_test_divide, extract_time, batch_generator


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


class Predictor():
    def __init__(self, seq_len, dim, hidden_dim, **kwargs): 
        super(Predictor, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.predictor = self.build_predictor()

        # self.model.compile(optimizer=Adam())
        self.loss_func = MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.predictor.compile(loss = self.loss_func, optimizer = self.optimizer, run_eagerly=True)


    def build_predictor(self,): 
        input_ =  Input(shape=(self.seq_len-1, self.dim-1), name='X_real_input')     
        x = GRU(units = self.hidden_dim, return_sequences=True, activation = 'tanh', name='p_gru', dtype='float64')(input_)
        x = TimeDistributed(layer = Dense(units = 1) )(x)
        output = tf.squeeze(x)
        model = Model(input_, output, name = 'predictor')
        return model


    def fit(self, X, y, epochs, verbose = 0, validation_split = None, shuffle = True):
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, min_delta = 1e-4, patience=50) 
        print_callback = PrintLossPerNthEpoch(print_period = 50)

        Y_hat = self.predictor.fit(X, y, 
            epochs = epochs, 
            verbose = verbose, 
            validation_split = validation_split,
            shuffle = shuffle,
            callbacks = [print_callback, early_stop_callback]
            )
        return Y_hat
        

    def __call__(self, X): 
        return self.predict(X)


    def predict(self, X): 
        Y_hat = self.predictor(X)
        return Y_hat


    def summary(self):
        self.predictor.summary()


def split_into_train_test(data, train_perc = 0.8): 
    N = data.shape[0]
    Ntrain = int(N * train_perc)
    indices = np.random.permutation(N)
    training_idx, test_idx = indices[:Ntrain], indices[Ntrain:]
    training, test = data[training_idx,:], data[test_idx,:]
    return training, test


def predictive_score_metrics (orig_data, generated_data, epochs = 2500):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
    - orig_data: original data
    - generated_data: generated synthetic data

    Returns:
    - predictive: mean abs error
    """
    # Basic Parameters
    no, seq_len, dim = orig_data.shape    

    # train on generated (synthetic) data, test on real data
    X_train, Y_train = generated_data[:, :-1, :-1], generated_data[:, 1:, -1]
    X_test, Y_test = orig_data[:, :-1, :-1], orig_data[:, 1:, -1]
    
    # shuffle both X and Y in unison
    X_train, Y_train = shuffle (X_train, Y_train)

    ## Builde a post-hoc RNN predictor network 
    # Network parameters
    hidden_dim = int(dim/2)    
    batch_size = 128

    predictor = Predictor(seq_len = seq_len, 
        dim = dim, 
        hidden_dim = int(dim/2)
        )

    predictor.fit(
        X_train, Y_train,
        epochs = epochs, 
        shuffle = True, 
        verbose=0,
        validation_split=0.1
    )
    
    y_test_hat = predictor.predict(X_test)
    predictive_score = mean_absolute_error(Y_test, y_test_hat )        
    return predictive_score 


        


if __name__== '__main__': 

    data= np.random.randn(100, 24, 5)
    X, Y = data[:, :-1, :-1], data[:, 1:, -1]

    N, T, D = data.shape

    predictor = Predictor(
        seq_len = T, 
        dim = D, 
        hidden_dim = int(D/2)
    )

    predictor.compile(optimizer = Adam(), loss='mse')

    predictor.summary()

    pred = predictor(X)

    print(pred.shape)
    print(Y.shape)

    # predictor.fit(
    #     X, Y,
    #     epochs =200, 
    #     verbose = 1
    # )

