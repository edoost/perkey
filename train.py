import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from data_loader import DataLoader
from common import config
from pr import precision_and_recall
from rouge import rouge


#tf.enable_eager_execution()
print('*** Tensorflow executing eagerly:', tf.executing_eagerly(), '\n')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

data_loader = DataLoader()

round_nums = 100
num_steps = 1000

beam_width = 50

SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

