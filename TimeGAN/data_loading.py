"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np


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


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data
  
  
  
def real_data_loading2(data_name, seq_len, per):
  """Load and preprocess the most recent real-world datasets.
     This is designed for Scenario-2
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    - per: percentage of observed data (e.g., 5%, 10%, 15%)
    
  Returns:
    - data: preprocessed data.
  """
    
  #assert data_name in ['stock','energy', 'new_sines']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'new_sines':
    ori_data = np.loadtxt('data/new_sines_data.csv', delimiter = ",")
  
        
  # We only get n% of recent data for scenario-2:
  num_samples = len(ori_data)
  p_rate = int(np.ceil(per * num_samples))
  new_ori_data = ori_data[-p_rate:]
  
  # Flip the data to make chronological data
  new_ori_data = new_ori_data[::-1]
  # Normalize the data
  new_ori_data = MinMaxScaler(new_ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(new_ori_data) - seq_len):
    _x = new_ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data
  
  
def real_data_loading_sce1(data_name, seq_len, test_per):
  """Load and preprocess the real-world datasets: Stock & Energy.
     This is designed for Scenario-1 with 5% as the testing set.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    - test_per: percentage of data as the testing data (5%)
    
  Returns:
    - train_data: preprocessed data.
    - test_data: preprocessed data.
  """
    
  #assert data_name in ['stock','energy', 'new_sines']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  
        
  # We only get n% of recent data for scenario-2:
  num_samples = len(ori_data)
  p_rate = int(np.ceil(test_per * num_samples))
  train_data = ori_data[:-p_rate] # training set
  test_data = ori_data[-p_rate:] # testing set
  
  
  # Flip the data to make chronological data
  train_data = train_data[::-1]
  test_data = test_data[::-1]
  # Normalize the data
  train_data = MinMaxScaler(train_data)
  test_data = MinMaxScaler(test_data)
    
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
  

def sine_data_generation_sce1(no, seq_len, dim, test_per):
  """Sine data generation.
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    - test_per: testing set percentage
  Returns:
    - train_data: generated data
    - test_data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
    
    p_rate = int(np.ceil(test_per * no))
    train_data = data[:-p_rate]
    test_data = data[-p_rate:]
                
  return train_data, test_data


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
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  
        
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
  # Normalize the data
  train_data = MinMaxScaler(train_data)
  test_data = MinMaxScaler(test_data)
    
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


def sine_data_generation_sce2(no, seq_len, dim, test_per, observe_per):
  """Sine data generation for Scenario-2.
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    - test_per: testing set percentage
  Returns:
    - train_data: generated data
    - test_data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
    
    p_rate = int(np.ceil(test_per * no))
    train_data = data[:-p_rate]
    test_data = data[-p_rate:] # testing data samples

    # select the most observe_per% of the training samples
    train_num_sample = len(train_data)
    ob_rate = int(np.ceil(observe_per * train_num_sample))
    train_data = train_data[-ob_rate:] # observed data samples

  return train_data, test_data