# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py

####
#this is a test for git
####

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
#K.set_floatx('float32')
# K.set_session(session)

from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam
from keras import regularizers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

#from scipy.misc import imread

np.random.seed(5)
tf.random.set_random_seed(3)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

batch_size = int(512 / 8)
class_weight ={0: 10, 1: 1}

def f1_acc(y_true, y_pred):

    # import numpy as np

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    # if K.isnan(f1_score):
    #     print('c1:', c1)
    #     print('c2:', c2)
    #     print('c3:', c3)

    return f1_score

#Load in the 384x512 data to train on
DataDir = '/home/admin/Desktop'
DataYes = h5py.File(os.path.join(DataDir, 'RawDataDeploy','YES_TRAIN.hdf5'), 'r')
DataNo = h5py.File(os.path.join(DataDir, 'RawDataDeploy','NO_TRAIN.hdf5'), 'r')

TrainYes = DataYes["train"]
TrainNo = DataNo["train"]
TestYes = DataYes["test"]
TestNo = DataNo["test"]
ValYes = DataYes["val"]
ValNo = DataNo["val"]

# Concatenate the no,yes crater chunks together to make cohesive training sets.
TrainData = np.concatenate((TrainNo,TrainYes), axis=0) #[:,:,:,np.newaxis]
TestData = np.concatenate((TestNo,TestYes), axis=0) #[:,:,:,np.newaxis]
ValData = np.concatenate((ValNo,ValYes), axis=0) #[:,:,:,np.newaxis]


# And make answer vectors
TrainAnswers = np.ones(len(TrainNo) + len(TrainYes))
TrainAnswers[:len(TrainNo)] = 0
TestAnswers = np.ones(len(TestNo) + len(TestYes))
TestAnswers[:len(TestNo)] = 0
ValAnswers = np.ones(len(ValNo) + len(ValYes))
ValAnswers[:len(ValNo)] = 0

# Make generators to stream them.
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
#test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)
#test_generator = test_datagen.flow(TestData, TestAnswers, batch_size=batch_size, seed=5)

#load the model
model = load_model('/home/admin/Desktop/Saved_CNNs/NFP_acc_FOV{}.h5'.format(150), custom_objects={'f1_acc': f1_acc})

Checkpoint1 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/NFP_actual_f1.h5', verbose=1, save_best_only=True, monitor='val_f1_acc')
Checkpoint2 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/NFP_actual_loss.h5', verbose=1, save_best_only=True, monitor='val_loss')
Checkpoint3 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/NFP_actual_acc.h5', verbose=1, save_best_only=True, monitor='val_acc')

#Fit the model
model.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=15,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[Checkpoint1, Checkpoint2, Checkpoint3],
                   class_weight=class_weight
                   )

high_acc = load_model('/home/admin/Desktop/Saved_CNNs/NFP_actual_acc.h5', custom_objects={'f1_acc': f1_acc})
high_f1 = load_model('/home/admin/Desktop/Saved_CNNs/NFP_actual_f1.h5', custom_objects={'f1_acc': f1_acc})
low_loss = load_model('/home/admin/Desktop/Saved_CNNs/NFP_actual_loss.h5', custom_objects={'f1_acc': f1_acc})




no_preds = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("high acc:")
print("no: ", x)
print("yes: ", y)
print(' ')

no_preds = high_f1.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = high_f1.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("high f1:")
print("no: ", x)
print("yes: ", y)
print(' ')

no_preds = low_loss.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = low_loss.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("low loss:")
print("no: ", x)
print("yes: ", y)
