#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings, os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from nbeats_keras.model import NBeatsNet as NBeatsKeras
warnings.filterwarnings(action='ignore', message='Setting attributes')
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.compat.v1.ConfigProto() # Another Version: config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[3]:


# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch

        
def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

        
# to load real data for Scenario-2:        
def real_data_loading_sce2(data_name, seq_len, test_per, observe_per):
  """Load and preprocess the real-world datasets: Stock & Energy.
     This is designed for Scenario-2 with 5% as the testing set.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    - test_per: percentage of data as the testing data (5%)
    - observe_per: percentage of data as the observed training samples (5% or 10%)
    
  Returns:
    - train_data: preprocessed data.
    - test_data: preprocessed data.
  """
    
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('/data/home/stufs1/zuwang/dg21/TimeGAN/data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('/data/home/stufs1/zuwang/dg21/TimeGAN/data/energy_data.csv', delimiter = ",",skiprows = 1)
  
        
  # We only get n% of recent data for scenario-2:
  num_samples = len(ori_data)
  p_rate = int(np.ceil(test_per * num_samples))
  train_data = ori_data[:-p_rate] # training set
  test_data = ori_data[-p_rate:] # testing set
  
  # get the number of observed data:
  train_num_sample = len(train_data)
  ob_rate = int(np.ceil(observe_per * train_num_sample))
  train_data = train_data[-ob_rate:] # observed data samples
  
  
  # Flip the data to make chronological data
  train_data = train_data[::-1]
  test_data = test_data[::-1]
  # Normalize the data -- will do it later
  # train_data = MinMaxScaler(train_data)
  # test_data = MinMaxScaler(test_data)
    
  # Preprocess the dataset
  temp_train_data = []
  temp_test_data = []    
  
  # Cut data by sequence length
  for i in range(0, len(train_data) - seq_len):
    _x = train_data[i:i + seq_len]
    temp_train_data.append(_x)

  for i in range(0, len(test_data) - seq_len):
    _y = test_data[i:i + seq_len]
    temp_test_data.append(_y)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_train_data))    
  new_train_data = []
  for i in range(len(temp_train_data)):
    new_train_data.append(temp_train_data[idx[i]])
  
  # Testing set does not need to permutate. 

  return new_train_data, temp_test_data


# In[6]:


# load the generated data -- Abu's data
src_path = "/data/home/stufs1/zuwang/dg21/TimeGAN/"
filename = "stock_data_TimeGAN_5000_sce2-wTest-0-010.npy"

seq_len = 24
syn_data = np.load(os.path.join(src_path, filename))
# syn_data = MinMaxScaler(syn_data)

'''
temp_data = []    
for i in range(0, len(syn_data) - seq_len):
    _x = syn_data[i:i + seq_len]
    temp_data.append(_x)

# Mix the datasets (to make it similar to i.i.d)
idx = np.random.permutation(len(temp_data))    
data = []
for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
'''

syn_data = np.asarray(syn_data)
print("synthetic data shape:", syn_data.shape) # (no, seq_len, dim)


# In[7]:


# load the real data -- TimeGAN
data_name = "stock"
seq_len = 24
test_per = 0.05
observe_per = 0.10
train_data, test_data = real_data_loading_sce2(data_name, seq_len, test_per, observe_per) # 
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
print("real train data shape:", train_data.shape) # (no, seq_len, dim)
print("real test data shape:", test_data.shape) # (no, seq_len, dim)

# In[8]:


len(syn_data[0])


# In[11]:


num_samples, time_steps, input_dim, output_dim = 1048551, 24, 6, 6
backend = NBeatsKeras(
        input_dim=input_dim,
        backcast_length=20, forecast_length=4,
        stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
        nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
        hidden_layer_units=64
    )

# Definition of the objective function and the optimizer.
backend.compile(loss='mae', optimizer='adam')

forecast_length = 4
backcast_length = 20

#### Stock-TimeGAN
# x: data backcast/y: forecast generation.


x_train, y_train = [], []
for i in range(len(syn_data)):
    x_train.append(syn_data[i][:20,:])
    y_train.append(syn_data[i][20:,:])

x_test, y_test = [], []
for i in range(len(test_data)):
    x_test.append(test_data[i][:20,:])
    y_test.append(test_data[i][20:,:])


'''
x_train, y_train = [], []
for i in range(len(syn_data)):
    for epoch in range(backcast_length, len(syn_data[i]) - forecast_length):
        x_train.append(syn_data[i][epoch - backcast_length:epoch])
        y_train.append(syn_data[i][epoch:epoch + forecast_length])

x_test, y_test = [], []
for i in range(len(ori_data)):
    for epoch in range(backcast_length, len(ori_data[i]) - forecast_length):
        x_test.append(ori_data[i][epoch - backcast_length:epoch])
        y_test.append(ori_data[i][epoch:epoch + forecast_length])
'''

# normalization.
norm_constant = np.max(x_train)
x_train, y_train = x_train / norm_constant, y_train / norm_constant
x_test, y_test = x_test / norm_constant, y_test / norm_constant
test_size = len(x_test)
print("test_size is:", test_size)

# check the model data shape
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# repeat 5 time of model training:
num_times = 5 
num_MAE = []
num_MSE = []

for idx in range(num_times):
  mdname = 'nBeats-TimeGAN-stock-best-sce2-wTest-0-010.h5'
  mdpath = "/data/home/stufs1/zuwang/dg21/n-beats/examples/models/"
  mdfile = os.path.join(mdpath, mdname)
  
  ckpt = ModelCheckpoint(mdfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #save whole model
  early = EarlyStopping(monitor='val_loss', mode='min', patience=10) # prevent overfitting
  
  # save log files
  logpath = "/data/home/stufs1/zuwang/dg21/n-beats/examples/logs/"
  logname = 'Log-' + "nBeats-TimeGAN-stock-sce2-wTest-0-010"  + '-.txt'
  logfile = os.path.join(logpath, logname)
  csv_logger = CSVLogger(logfile, append=True)
  reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
  callbacks_list = [ckpt, early, csv_logger, reduceLR]
  
  
  # Train the model.
  print('Training...')
  #backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=128, callbacks=callbacks_list)
  backend.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=128, shuffle=True, callbacks=callbacks_list)
  
  
  # Save the model for later.
  backend.save('n_beats_model_abu_stock.h5')
  
  # Predict on the testing set (forecast).
  predictions_forecast = backend.predict(x_test)
  print("the prediction_forecast shape is:", predictions_forecast.shape) #shape: (30, 5, 1)
  #np.testing.assert_equal(predictions_forecast.shape, (test_size, backend.forecast_length, output_dim))
  
  # Predict on the testing set (backcast).
  predictions_backcast = backend.predict(x_test, return_backcast=True)
  print("the prediction_backcast shape is:", predictions_backcast.shape) #shape: (30, 15, 1)
  #np.testing.assert_equal(predictions_backcast.shape, (test_size, backend.backcast_length, output_dim))
  
  
  # Load the model.
  src_path = "/data/home/stufs1/zuwang/dg21/n-beats/examples/models/"
  modelname = os.path.join(src_path, "nBeats-TimeGAN-stock-best-sce2-wTest-0-010.h5")
  model_2 = NBeatsKeras.load(modelname)
  predicts = model_2.predict(x_test)
  print("the reloaded prediction_shape is:", predicts.shape) #shape: (30, 5, 1)
  #np.testing.assert_almost_equal(predictions_forecast, model_2.predict(x_test))
  
  num_sample = len(x_test)
  
  MAE_temp = 0.0
  MSE_temp = 0.0
  
  for i in range(num_sample):
      MAE_temp = MAE_temp + mean_absolute_error(y_test[i], predicts[i])
      MSE_temp = MSE_temp + mean_squared_error(y_test[i], predicts[i])

  predictive_score_mae = MAE_temp/num_sample
  predictive_score_mse = MSE_temp/num_sample
  print("predictive_score_mae:", predictive_score_mae)
  print("predictive_score_mse:", predictive_score_mse)
  num_MAE.append(predictive_score_mae)
  num_MSE.append(predictive_score_mse)

print("All the predictive MAE scores: ", num_MAE)
print('Average Predictive MAE score: ' + str(np.round(np.mean(num_MAE), 4)))
print('Average Predictive MSE score: ' + str(np.round(np.mean(num_MSE), 4)))


subplots = [221, 222, 223, 224]
plt.figure(1)
for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
    p1 = np.expand_dims(predicts[i][:,0], axis=-1)
    x1 = np.expand_dims(x_test[i][:,0], axis=-1)
    y1 = np.expand_dims(y_test[i][:,0], axis=-1)
    ff, xx, yy = p1 * norm_constant, x1 * norm_constant, y1 * norm_constant
    plt.subplot(subplots[plot_id])
    plt.grid()
    plot_scatter(range(0, backcast_length), xx, color='b')
    plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
    plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
plt.savefig("nbeats-predictions-stocks.png", dpi=300)
plt.show()





