
import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, TimeDistributed, Reshape, Input, Layer, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, Mean
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.callbacks import Callback, EarlyStopping


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


class ConvPredictor:    
    def __init__(self,  seq_len, feat_dim, hidden_layer_sizes ):
        self.seq_len = seq_len
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feat_dim = feat_dim

        self.predictor = self.build_predictor()

        self.loss_func = MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.predictor.compile(loss = self.loss_func, optimizer = self.optimizer)


    def build_predictor(self):
        input_ = Input(shape=(self.seq_len, self.feat_dim), name='input')
        x = input_
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                    # filters = num_filters, 
                    filters = self.feat_dim//2, 
                    kernel_size=3, 
                    activation='tanh', 
                    padding='causal',
                    name=f'conv_{i}')(x)

        x = TimeDistributed(layer = Dense(units = self.feat_dim//2, activation='relu') )(x)
        x = TimeDistributed(layer = Dense(units = 1) )(x)
        output = Flatten(name='flatten')(x)
        model = Model(input_, output, name = 'predictor')
        return model


    def fit(self, X, y, epochs, verbose = 0, validation_split = None, shuffle = True, print_period=50):
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, min_delta = 1e-4, patience=50) 
        print_callback = PrintLossPerNthEpoch(label = 'Pred', print_period = print_period)

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



#####################################################################################################
#####################################################################################################




if __name__== '__main__': 
    N, T, D = 100, 24, 5
    data= np.random.randn(N, T, D)
    pred_len = 1
    X, Y = data[:, :-pred_len, :-1], data[:, pred_len:, -1]

    print('orig shapes', X.shape, Y.shape)

    N_x, T_x, D_x = X.shape

    predictor = Predictor(
        seq_len = T_x, 
        feat_dim = D_x, 
        hidden_layer_sizes = [50, 100]
    )

    predictor.summary()

    pred = predictor(X)

    print('pred shape: ', pred.shape)

    # predictor.fit(
    #     X, Y,
    #     epochs =200, 
    #     verbose = 1
    # )



