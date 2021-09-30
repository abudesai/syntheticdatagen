
# Necessary Packages
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, Conv1D, GRU, Flatten, Dense, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, Mean
from tensorflow.keras.callbacks import Callback, EarlyStopping
import sys 
import nbeats_model
import predictor as conv_pred



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


class Predictor():
    def __init__(self, seq_len, dim, hidden_dim, **kwargs): 
        super(Predictor, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.predictor = self.build_predictor()

        self.loss_func = MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.predictor.compile(loss = self.loss_func, optimizer = self.optimizer)


    def build_predictor(self,): 
        input_ =  Input(shape=(self.seq_len-1, self.dim-1), name='X_real_input')     
        x = GRU(units = self.hidden_dim, return_sequences=True, activation = 'tanh', name='p_gru')(input_)
        x = TimeDistributed(layer = Dense(units = 1) )(x)
        output = tf.squeeze(x)
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
        print_callback = PrintLossPerNthEpoch(label ='Pred', print_period = print_period)

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

def predictive_score_metrics (orig_data, generated_data, epochs = 2500, predictor = 'conv', print_epochs = 50):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
    - orig_data: original data
    - generated_data: generated synthetic data

    Returns:
    - predictive: mean abs error
    """
    global print_period
    print_period = print_epochs

    # Basic Parameters
    no, seq_len, dim = orig_data.shape   

    # --------------------------------------------------------------------------
    # nbeats 

    if predictor == 'nbeats':         

        # train on generated (synthetic) data, test on real data
        # X_train, E_train = generated_data[:, :-1, -1:], generated_data[:, :-1, :-1]
        # Y_train = generated_data[:, 1:, -1]
        # X_test, E_test = orig_data[:, :-fcst_len, :1], orig_data[:, :-fcst_len, 1:]
        # Y_test = orig_data[:, -fcst_len:, -1]

        fcst_len = 5
        X_train, E_train = generated_data[:, :-fcst_len, -1:], generated_data[:, :-fcst_len, :-1]
        Y_train = generated_data[:, -fcst_len:, -1]
        X_test, E_test = orig_data[:, :-fcst_len, -1:], orig_data[:, :-fcst_len, :-1]
        Y_test = orig_data[:, -fcst_len:, -1]
        

        ## Builde a post-hoc predictor network 
        # Network parameters
        N, T, D = X_train.shape

        predictor = nbeats_model.NBeatsNet(
            input_dim = D, 
            exo_dim = E_train.shape[2],
            backcast_length = X_train.shape[1],
            forecast_length = Y_train.shape[1],
            nb_blocks_per_stack=2,
            stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
            thetas_dim=(16, 16),
            hidden_layer_units = 64, 
        )

        predictor.compile(loss='mse', optimizer='adam')

        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta = 1e-5, patience=30) 
        print_callback = PrintLossPerNthEpoch(print_period = print_period)

        predictor.fit(
            [X_train, E_train], Y_train, epochs = epochs, verbose = 0, validation_split= 0.1,
            callbacks = [early_stop_callback, print_callback]
        )

        y_test_hat = predictor.predict([X_test,  E_test])
        y_test_hat = np.squeeze(y_test_hat)
        # print('predict shape: ', y_test_hat.shape)

        y_test_hat = y_test_hat.reshape((-1, 1))
        Y_test = Y_test.reshape((-1, 1))

    else:

        no, seq_len, dim = orig_data.shape    

        # train on generated (synthetic) data, test on real data
        X_train, Y_train = generated_data[:, :-1, :-1], generated_data[:, 1:, -1]
        X_test, Y_test = orig_data[:, :-1, :-1], orig_data[:, 1:, -1]

        # X_train, Y_train = generated_data[:, :-1, 1:], generated_data[:, 1:, 0]
        # X_test, Y_test = orig_data[:, :-1, 1:], orig_data[:, 1:, 0]
        
        # shuffle both X and Y in unison
        X_train, Y_train = shuffle (X_train, Y_train)

        if predictor == 'rnn': 
            ## Builde a post-hoc RNN predictor network 
            # Network parameters
            hidden_dim = int(dim/2)    
            batch_size = 128
            predictor = Predictor(seq_len = seq_len, 
                dim = dim, 
                hidden_dim = int(dim/2)
                )
                
        elif predictor == 'conv': 
            predictor = ConvPredictor(
                seq_len = seq_len-1, 
                feat_dim = dim-1, 
                hidden_layer_sizes = [50,  100],
                )

        predictor.fit(
            X_train, Y_train,
            epochs = epochs, 
            shuffle = True, 
            verbose=0, 
            validation_split=0.1, 
            print_period = print_period
        )
        y_test_hat = predictor.predict(X_test)


    predictive_score = mean_absolute_error(Y_test, y_test_hat )   
    # predictive_score = mean_squared_error(Y_test, y_test_hat )   
    return predictive_score 


        


if __name__== '__main__': 

    N, T, D = 100, 24, 5
    data= np.random.randn(N, T, D)
    pred_dims = 1

    fcst_len = 5

    X, E = data[:, :-fcst_len, -1:], data[:, :-fcst_len, :-1]
    Y = data[:, -fcst_len:, -1]

    print('orig shapes', X.shape, E.shape, Y.shape)

    N_x, T_x, D_x = X.shape

    # predictor = Predictor(
    #     seq_len = T_x, 
    #     feat_dim = D_x, 
    #     hidden_layer_sizes = [50, 100]
    # )

    predictor = NBeatsNet(
        input_dim = X.shape[2],
        exo_dim = E.shape[2],
        backcast_length = X.shape[1],
        forecast_length = Y.shape[1],
    )

    predictor.summary()

    pred = predictor.predict([X, E])

    print('pred shape: ', pred.shape)

    # predictor.fit(
    #     X, Y,
    #     epochs =200, 
    #     verbose = 1
    # )

