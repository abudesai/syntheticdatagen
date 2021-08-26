
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
 

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,  
            seq_len, 
            feat_dim,  
            latent_dim,
            reconstruction_wt = 3.0,
            **kwargs  ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean( name="reconstruction_loss" )
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.encoder = None
        self.decoder = None


    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        try: 
            # this is was only required during testing when working with different data shapes
            # throw out this try/except block later...
            if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        except: pass
        return x_decoded


    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams


    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    
    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()


    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            # reconstruction, a, b, c = self.decoder(z)  # used during testing
            reconstruction = self.decoder(z)

            err = tf.math.squared_difference(X, reconstruction)
            reconstruction_loss = tf.reduce_sum(err)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def save_weights(self, model_dir, file_pref): 
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    
    def load_weights(self, model_dir, file_pref):
        encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)



#####################################################################################################
#####################################################################################################


if __name__ == '__main__':

    pass