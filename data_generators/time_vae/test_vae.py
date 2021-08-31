
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

from random import shuffle
import sys
from datetime import datetime, timedelta
import numpy as np , pandas as pd
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as VAE_ConvI
import utils



if __name__ == '__main__':
    start = time.time()
    
    data_dir = "../../data/processed_orig_data/"
    # ----------------------------------------------------------------------------------
    
    dataset = 'sine'            # sine, stocks, energy
    perc = 100                    # 5, 10, 20, 100    
    valid_perc = 0.0
    vae_type = 'vae_conv_I'           # vae_dense, vae_conv, vae_convI
    # ----------------------------------------------------------------------------------
    # read data
    input_file = f'{dataset}_subsampled_train_perc_{perc}.npz'
    full_train_data = utils.get_training_data(data_dir + input_file)
    N, T, D = full_train_data.shape   
    print('data shape:', N, T, D) 

    # ----------------------------------------------------------------------------------
    # further split the training data into train and validation set - same thing done in forecasting task
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train

    # Shuffle data
    np.random.shuffle(full_train_data)

    train_data = full_train_data[:N_train]
    valid_data = full_train_data[N_train:]
    print("train/valid shapes: ", train_data.shape, valid_data.shape)    
    # sys.exit()
    
    # ----------------------------------------------------------------------------------
    # min max scale the data
    # scaler = utils.MinMaxScaler_Feat_Dim( scaling_len = T, input_dim = D, upper_bound = 3.0, lower_bound = -3.0 ) 
    scaler = utils.MinMaxScaler()      
    scaled_train_data = scaler.fit_transform(train_data)

    scaled_valid_data = scaler.transform(valid_data)
    # joblib.dump(scaler, 'scaler.save')  
    # print("train/valid shapes: ", scaled_train_data.shape, scaled_valid_data.shape)
    # sys.exit()

    # ----------------------------------------------------------------------------------
    # instantiate the model     
    
    latent_dim = 8 #min( 2*D, 2 + D // 3)

    if vae_type == 'vae_dense': 
        vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
    elif vae_type == 'vae_conv':
        vae = VAE_Conv( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[100, 200] )
    elif vae_type == 'vae_conv_I':
        vae = VAE_ConvI( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[50, 100, 200],  
            reconstruction_wt = 3.0,
            # trend_poly=1, 
            # num_gen_seas=1,
            # custom_seas = [ (T//2, 1)] ,     # list of tuples of (num_of_seasons, len_per_season)
            use_residual_conn = True
            )
    else:  raise Exception('wut')
    
    vae.compile(optimizer=Adam())
    # vae.summary() ; sys.exit()
    # ----------------------------------------------------------------------------------

    early_stop_loss = 'loss'
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=50) 
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)
    # sys.exit()

    print("Running training...")
    r = vae.fit(
        scaled_train_data, 
        batch_size = 32,
        epochs=100,
        shuffle = True,
        callbacks=[early_stop_callback, reduceLR],
        verbose = 0
    )

    print("Final losses: ")
    # sys.exit()    
    # ----------------------------------------------------------------------------------

    # save model 
    # model_dir = './model/'
    # pref = f'vae_{vae_type}_{dataset}_perc_{perc}_iter_{0}_'
    # vae.save(model_dir, pref)
        

    # # ----------------------------------------------------------------------------------
    # visually check reconstruction 
    X = scaled_train_data
    # X = scaled_valid_data

    x_decoded = vae.predict(scaled_train_data)
    print('x_decoded.shape', x_decoded.shape)

    ### compare original and posterior predictive (reconstructed) samples
    utils.draw_orig_and_post_pred_sample(X, x_decoded, n=5)    

    # # Plot the prior generated samples over different areas of the latent space
    if latent_dim == 2: utils.plot_latent_space_timeseries(vae, n=8, figsize = (20, 10))
        
    # # ----------------------------------------------------------------------------------
    # draw random prior samples
    num_samples = N_train
    # print("num_samples: ", num_samples)

    samples = vae.get_prior_samples(num_samples=num_samples)
    
    utils.plot_samples(samples, n=5)

    # inverse-transform scaling 
    samples = scaler.inverse_transform(samples)
    print('shape of gen samples: ', samples.shape) 

    print('-'*80)
    print('orig scale orig data means: ', train_data.mean(axis=0).mean(axis=1))
    print('orig scale gen data means: ', samples.mean(axis=0).mean(axis=1))
    print('-'*80)
    print('scaled orig data means: ', scaled_train_data.mean(axis=0).mean(axis=1))    
    print('scaled decoded data means: ', x_decoded.mean(axis=0).mean(axis=1))    
    print('-'*80)
    # ----------------------------------------------------------------------------------

    # save samples
    output_dir = "../../data/generated_data/"
    samples_fpath = f'{vae_type}_gen_samples_{dataset}_perc_{perc}.npz'  
    np.savez_compressed(os.path.join( output_dir, samples_fpath), data=samples)

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # later.... load model 
    # new_vae = VAE_ConvI.load(model_dir, pref)

    # new_x_decoded = new_vae.predict(scaled_train_data)
    # print('new_x_decoded.shape', new_x_decoded.shape)

    # print('Preds from orig and loaded models equal: ', np.allclose( x_decoded,  new_x_decoded, atol=1e-5))
    
    # sys.exit()
    

    end = time.time()
    print(f"Total run time: {np.round((end - start)/60.0, 2)} minutes") 