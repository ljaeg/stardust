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

model = ResNet50(classes = 1, input_shape = (500, 500, 1), pooling = 'max', weights = None)

# TrainNo = np.array(DataFile['TrainNo'])
# TrainYes = np.array(DataFile['TrainYes'])
# TestNo = np.array(DataFile['TestNo'])
# TestYes = np.array(DataFile['TestYes'])
# ValNo = np.array(DataFile['ValNo'])
# ValYes = np.array(DataFile['ValYes'])

DataDir = '/home/admin/Desktop'
DataFile = h5py.File(os.path.join(DataDir, 'Aug6','to_train_500.hdf5'), 'r+')
# TestYes_500 = DF1['TestYes']
# TestNo_500 = DF1['TestNo']
# y_500 = model.predict(np.reshape(TestYes_500, (len(TestYes_500), 500, 500, 1)))
# n_500 = model.predict(np.reshape(TestNo_500, (len(TestNo_500), 500, 500, 1)))
# b4_y = len([i for i in y_500 if i > .5]) / len(y_500)
# b4_n = len([i for i in n_500 if i < .5]) / len(n_500)
# print('500x500 w craters:')
# print(b4_y)
# print('500x500 no craters:')
# print(b4_n)
# print(' ')

TrainNo = np.array(DataFile['TrainNo'])
TrainYes = np.array(DataFile['TrainYes'])
TestNo = np.array(DataFile['TestNo'])
TestYes = np.array(DataFile['TestYes'])
ValNo = np.array(DataFile['ValNo'])
ValYes = np.array(DataFile['ValYes'])

batch_size = 16

# Concatenate the no,yes crater chunks together to make cohesive training sets.
TrainData = np.concatenate((TrainNo,TrainYes), axis=0)[:,:,:,np.newaxis]
TestData = np.concatenate((TestNo,TestYes), axis=0)[:,:,:,np.newaxis]
ValData = np.concatenate((ValNo,ValYes), axis=0)[:,:,:,np.newaxis]


# And make answer vectors
TrainAnswers = np.ones(len(TrainNo) + len(TrainYes))
TrainAnswers[:len(TrainNo)] = 0
TestAnswers = np.ones(len(TestNo) + len(TestYes))
TestAnswers[:len(TestNo)] = 0
ValAnswers = np.ones(len(ValNo) + len(ValYes))
ValAnswers[:len(ValNo)] = 0

# Make generators to stream them.
train_datagen = ImageDataGenerator(zca_whitening = True)
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)
test_generator = test_datagen.flow(TestData, TestAnswers, batch_size=batch_size, seed=5)

#model.compile(optimizer=Nadam(lr=0.0002), loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.compile(optimizer=Nadam(lr=0.0002), loss='binary_crossentropy', metrics=['acc'])
model.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=30,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   )





