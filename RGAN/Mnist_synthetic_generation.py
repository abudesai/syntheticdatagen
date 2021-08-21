import numpy as np
import tensorflow as tf
import pdb, os
import random
import json
from scipy.stats import mode

import data_utils
import plotting
import model
import utils
import eval

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

print("All imports worked")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.compat.v1.ConfigProto() # Another Version: config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.logging.set_verbosity(tf.logging.ERROR)

# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: 
  settings = utils.load_settings_from_file(settings)
  print("==========================still running here.==========================")
  print("the current seeting_file is:", settings['settings_file'])
print('Ready to run with settings:')

epoch = 450
num_samples = 4 #10
idx="Test"
seq_length=settings["seq_length"]
labs = np.array([0,1,2,3,4,5,6,7,8,9])

'''
csamples = np.zeros((10,10))

csamples[0][0] = 1
csamples[1][1] = 1
csamples[2][2] = 1
csamples[3][3] = 1
csamples[4][4] = 1
csamples[5][5] = 1
csamples[6][6] = 1
csamples[7][7] = 1
csamples[8][8] = 1
csamples[9][9] = 1
'''

csamples = np.zeros((4,10))

csamples[0][5] = 1
csamples[1][3] = 1
csamples[2][0] = 1
csamples[3][9] = 1

print(csamples)

synth_data = model.sample_trained_model(settings, epoch, num_samples, C_samples=csamples)
# synth_data.shape: (4, 28, 28)
plotting.save_mnist_plot_sample(synth_data.reshape(-1, seq_length**2, 1), idx,"epoch450", num_samples, labels=labs)