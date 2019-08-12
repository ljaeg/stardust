import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
#os.environ['KERAS_BACKEND'] = 'theano'

# Tell tensorflow to only use two CPUs so I can use my computer for other stuff too.
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
# config = tf.ConfigProto(intra_op_parallelism_threads=2)
# session = tf.Session(config=config)
import keras.backend as K
# K.set_session(session)

from keras.models import Sequential, load_model, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam
from keras import regularizers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from keras.applications.resnet50 import ResNet50 

model = ResNet50(classes = 1, input_shape = (None, None, 1), pooling = 'max')

DataDir = '/home/admin/Desktop'
DF1 = h5py.File(os.path.join(DataDir, 'Aug6','to_train_500.hdf5'), 'r+')
TestYes_500 = DF1['TestYes']
TestNo_500 = DF1['TestNo']
y_500 = model.predict(np.reshape(TestYes_500, (len(TestYes_500), 500, 500, 1)))
n_500 = model.predict(np.reshape(TestNo_500, (len(TestNo_500), 500, 500, 1)))
b4_y = len([i for i in y_500 if i > .5]) / len(y_500)
b4_n = len([i for i in n_500 if i < .5]) / len(n_500)
print('500x500 w craters:')
print(b4_y)
print('500x500 no craters:')
print(b4_n)
print(' ')