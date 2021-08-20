
import os, random, warnings
import numpy as np , pandas as pd
from datetime import datetime, timedelta
import joblib, h5py
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model 

from vae_dense_model import VariationalAutoencoderDense as VAE
# from vae_conv_model import VariationalAutoencoderConv as VAE

from preprocess_pipeline import get_preprocess_pipelines
from config import config as cfg

DEBUG = False


class VAE_Wrapper:

    MIN_HISTORY_LEN = 60        # in days

    def __init__(self, 
                encode_len, 
                decode_len, 
                latent_dim,
                hidden_layer_sizes,
                loss_decay_const = 0.99,
                reconstruction_wt=5.0,
                rand_seed = None):
        
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.latent_dim = latent_dim
        self.loss_decay_const = loss_decay_const
        self.hidden_layer_sizes = hidden_layer_sizes
        self.reconstruction_wt = reconstruction_wt

        if type(rand_seed) == int or type(rand_seed) == float:
            self.rand_seed = rand_seed
        else: 
            self.rand_seed = None
        if self.rand_seed is not None: self._set_seeds(self.rand_seed)

        self.vae_model = VAE(
            encode_len=encode_len,
            decode_len=(encode_len if decode_len == 'auto' else decode_len),
            latent_dim = latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            reconstruction_wt = reconstruction_wt
        )   

        self.vae_model.compile(optimizer=Adam()) 

        
        self.training_prep_pipeline, self.prediction_prep_pipeline = get_preprocess_pipelines(
                encode_len = self.encode_len,
                decode_len = self.decode_len
            )


    def _set_seeds(self, seed_value):   
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
    

        
    def assert_enough_data(self, input_data):
        min_date = input_data[cfg.TIME_COL].min()
        max_date = input_data[cfg.TIME_COL].max()
        history_len = (max_date - min_date).days + 1
        assert history_len >= self.MIN_HISTORY_LEN, f"Error: Insufficient history provided to fit model. Need at least {self.MIN_HISTORY_LEN} days. Provided {history_len} days."

        
    # def _preprocess_data(self, data, with_train_steps, is_prediction = False):
        
    #     preprocess_pipeline = get_preprocess_pipeline(
    #             with_train_steps = with_train_steps,
    #             encode_len = self.encode_len,
    #             decode_len = 0 if is_prediction == True else self.decode_len,
    #             shuffle = True, 
    #         )
    #     data = preprocess_pipeline.fit_transform(data)
    #     X = data['X']; Y = data['Y']
    #     return X, Y, preprocess_pipeline

    
    def fit(self, training_data, validation_split=0.1, verbose=0, batch_size = 128, max_epochs=2000):

        self.assert_enough_data(training_data)

        if self.rand_seed is not None: self._set_seeds(self.rand_seed)
                
        if DEBUG: print("Running main training ...")   

        # X, Y, _ = self._preprocess_data(training_data, with_train_steps=True) 

        processed_data_dict = self.training_prep_pipeline.fit_transform(training_data)
        X = processed_data_dict['X']
        Y = processed_data_dict['Y']

        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        if X.shape[0] < 50:  validation_split = None

        loss_to_monitor = 'loss' 
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, min_delta = 1e-2, patience=5) 

        history = self.vae_model.fit(
                X, Y, 
                epochs=max_epochs,  
                batch_size=batch_size,
                verbose = verbose,  
                callbacks=[early_stop_callback],
            )
        return history
        
    
    def get_prior_samples(self, Z):
        samples = self.vae_model.get_prior_samples(Z)
        return samples


    def save(self, file = lambda s: s):        
        joblib.dump(self.training_prep_pipeline, file(cfg.TRAIN_PIPE_FILE))
        joblib.dump(self.prediction_prep_pipeline, file(cfg.PRED_PIPE_FILE))

        encoder_wts = self.vae_model.encoder.get_weights()
        decoder_wts = self.vae_model.decoder.get_weights()
        joblib.dump(encoder_wts, file(cfg.ENCODER_WEIGHTS))
        joblib.dump(decoder_wts, file(cfg.DECODER_WEIGHTS))
        
        dict_params = {
            'encode_len': self.encode_len,
            'decode_len': self.decode_len,
            'latent_dim': self.latent_dim,
            'loss_decay_const': self.loss_decay_const,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'reconstruction_wt': self.reconstruction_wt,
            'rand_seed': self.rand_seed,
        }
        joblib.dump(dict_params, file(cfg.PARAMS_FILE))


    
    @staticmethod
    def load(file = lambda s: s):
        dict_params = joblib.load(file(cfg.PARAMS_FILE))
        model = VAE_Wrapper(
            encode_len = dict_params['encode_len'], 
            decode_len = dict_params['decode_len'],        # 'auto' if auto-encoding, or int > 0 if forecasting
            latent_dim = dict_params['latent_dim'], 
            hidden_layer_sizes=dict_params['hidden_layer_sizes'], 
            loss_decay_const = dict_params['loss_decay_const'], 
            reconstruction_wt = dict_params['reconstruction_wt'], 
            rand_seed = dict_params['rand_seed'], 
        )

        model.training_prep_pipeline = joblib.load(file(cfg.TRAIN_PIPE_FILE))
        model.prediction_prep_pipeline = joblib.load(file(cfg.PRED_PIPE_FILE))

        model.vae_model = VAE(
            encode_len = dict_params['encode_len'], 
            decode_len = dict_params['decode_len'],        # 'auto' if auto-encoding, or int > 0 if forecasting
            latent_dim = dict_params['latent_dim'], 
            hidden_layer_sizes=dict_params['hidden_layer_sizes'], 
            loss_decay_const = dict_params['loss_decay_const'], 
            reconstruction_wt = dict_params['reconstruction_wt'], 
        )

        encoder_wts = joblib.load(file(cfg.ENCODER_WEIGHTS))
        decoder_wts = joblib.load(file(cfg.DECODER_WEIGHTS))
        model.vae_model.encoder.set_weights(encoder_wts)
        model.vae_model.decoder.set_weights(decoder_wts)
        
        model.vae_model.compile(optimizer=Adam())

        return model


    def get_num_trainable_variables(self):
        return self.vae_model.get_num_trainable_variables()

    def predict(self, X_forecast):
        # preprocess data
        # X, Y, pipe = self._preprocess_data(X_forecast, with_train_steps=False, is_prediction = True) 

        processed_data_dict = self.prediction_prep_pipeline.transform(X_forecast)
        X = processed_data_dict['X']
        Y = processed_data_dict['Y']    

        # get original index and last date in history - need for prediction dataframe
        last_hist_date = X_forecast[cfg.TIME_COL].max()   
        orig_idx = X.index
        
        # # make predictions
        x_decoded = self.vae_model.predict(X)

        # transform numpy array of preds to dataframe 
        preds_df = self._transform_preds_as_df(x_decoded, orig_idx, last_hist_date)
        return preds_df


    def _transform_preds_as_df(self, preds, orig_idx, last_hist_date):
    
        # rescale data 
        preds = self.prediction_prep_pipeline[cfg.MINMAX_SCALER].inverse_transform(preds)
        preds = np.squeeze(preds)
        if len(preds.shape) == 1: preds = preds.reshape((1, -1))

        if self.decode_len == 'auto':
            all_time_ints = reversed( [ last_hist_date + timedelta(days=-s) for s in range(preds.shape[1]  )])
        else: 
            all_time_ints = [last_hist_date + timedelta(days=1+s) for s in range(preds.shape[1]  )]

        preds_df = pd.DataFrame(preds, columns=all_time_ints, index=orig_idx)
        preds_df = self.prediction_prep_pipeline[cfg.TIME_PIVOTER].inverse_transform(preds_df)

        return preds_df


    def summary(self): self.model.summary()




