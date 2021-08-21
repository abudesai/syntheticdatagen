#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

import tensorflow as tf

import os


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.compat.v1.ConfigProto() # Another Version: config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[6]:


## Data loading
data_name = 'sine'
seq_len = 24

if data_name in ['stock', 'energy']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)
    
print(data_name + ' dataset is ready.')


# In[7]:


## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 5000
parameters['batch_size'] = 128


# In[8]:


# scenario2-(1): 10% of the original data samples
# energy shape: (10000, 24, 6)

num_samples = len(ori_data)
p_rate = int(np.ceil(0.1 * num_samples)) # 10% of original samples
ori_data = np.array(ori_data)
new_ori_data = ori_data[:p_rate, :, :] #(1000, 24, 28)
print("new 10% data shape: ", new_ori_data.shape)
new_ori_data = list(new_ori_data)


# In[ ]:


# Run TimeGAN
generated_data = timegan(new_ori_data, parameters)   
print('Finish Synthetic Data Generation')


# In[ ]:


filename = data_name + "_data_TimeGAN_5000_sce2-1.npy"
np.save(filename, generated_data)


# In[ ]:


generated_data.shape


# In[ ]:


generated_data[0][0]


# In[ ]:


metric_iteration = 5

predictive_score = list()
for tt in range(metric_iteration):
  temp_pred = predictive_score_metrics(ori_data, generated_data)
  predictive_score.append(temp_pred)   
    
print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))


# In[ ]:


visualization(ori_data, generated_data, 'pca')
visualization(ori_data, generated_data, 'tsne')


# In[ ]:




