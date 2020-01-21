import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
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

from Reduce import reduce_whole_ds

#from scipy.misc import imread

np.random.seed(5)
tf.random.set_random_seed(3)

# Train/validate/test info
batch_size=int(512/16)
class_weight={0: 1, 1: 1}
epochs = 100
ConvScale=32
DenseScale=64 / 4
# GN1 = .054
# GN2 = .018
# GN3 = .14
# alpha = .24
spatial_d_rate = 0.15
GN1 = 0
GN2 = 0
GN3 = 0
alpha = 0
dropout_rate = 0.25
reg_scale = 0.0001
kernel_size = 3

# Calculate the F1 score which we use for optimizing the CNN.
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

def lr_schedule(epoch):
  orig = .01
  mult = .25 
  e = 2.7
  return orig*(e**(mult*epoch))


# Load the image datasets from the HDF.
# RunDir = '/home/zack/Data/SAH/Code/Gen002/001 - CNN'
# DataDir = '/home/zack/Data/SAH/Code/Gen002/Data'

DataDir = '/home/admin/Desktop'
DataFile = h5py.File(os.path.join(DataDir, 'Aug6','new_to_train_300.hdf5'), 'r+')

# DataDir = '/home/admin/Desktop'
# DataFile = h5py.File(os.path.join(DataDir, 'Aug6','Middle_FOV150_Num10k_new.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
try:
  Foils = DataFile.attrs['Foils'].split(',')
except:
  Foils = DataFile.attrs['Foils']
# Read the Train/Test/Val datasets.
num_ims = int(NumFOVs)
ad_sub = 0
TrainNo = np.array(DataFile['TrainNo'])
TrainYes = np.array(DataFile['TrainYes'])
TestNo = np.array(DataFile['TestNo'])
TestYes = np.array(DataFile['TestYes'])
ValNo = np.array(DataFile['ValNo'])
ValYes = np.array(DataFile['ValYes'])
print('before:', len(TrainNo))


block = (2, 2)
function = np.max
TrainNo = reduce_whole_ds(TrainNo, block_size = block, f = function)
TrainYes = reduce_whole_ds(TrainYes, block_size = block, f = function)
TestNo = reduce_whole_ds(TestNo, block_size = block, f = function)
TestYes = reduce_whole_ds(TestYes, block_size = block, f = function)
ValNo = reduce_whole_ds(ValNo, block_size = block, f = function)
ValYes = reduce_whole_ds(ValYes, block_size = block, f = function)
FOVSize = ValYes.shape[1]



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
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)
test_generator = test_datagen.flow(TestData, TestAnswers, batch_size=batch_size, seed=5)

# Define the NN
# Now define the neural network.
input_shape = (FOVSize, FOVSize, 1) # Only one channel since these are B&W.

model = Sequential()
model.add(GaussianNoise(GN1, input_shape = (None, None, 1)))
model.add(Conv2D(int(4*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
# model.add(GaussianNoise(GN2))
model.add(Conv2D(int(4*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))

model.add(GaussianNoise(GN3))
model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
# model.add(GaussianNoise(GN3))
model.add(Conv2D(int(ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D(pool_size = 2))
model.add(Conv2D(int(ConvScale), (int(kernel_size), int(kernel_size)), padding = 'valid', activation = 'relu', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(SpatialDropout2D(spatial_d_rate))

model.add(GlobalMaxPooling2D())
#model.add(Flatten())
model.add(Dense(int(2*DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(2*DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(2*DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Nadam(lr=0.0002), loss='binary_crossentropy', metrics=['acc', f1_acc])
model.save('/home/admin/Desktop/Saved_CNNs/Foils_CNN_FOV{}.h5'.format(FOVSize))
model = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_FOV{}.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})
model.summary()
# plot_model(model, to_file='Foils_CNN.png', show_shapes=True)



# Do the training
# CSVLogger is a checkpoint function.  After each epoch, it will write the stats from that epoch to a csv file.
Logger = CSVLogger('/home/admin/Desktop/Saved_CNNs/Foils_CNN_Log_FOV{}.txt'.format(FOVSize), append=True)
# ModelCheckpoint will save the configuration of the network after each epoch.
# save_best_only ensures that when the validation score is no longer improving, we don't overwrite
# the network with a new configuration that is overfitting.
Checkpoint1 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_F1_FOV{}.h5'.format(FOVSize), verbose=1, save_best_only=True, monitor='val_f1_acc')#'val_acc')
Checkpoint2 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_loss_FOV{}.h5'.format(FOVSize), verbose=1, save_best_only=True, monitor='val_loss')#'val_acc')
Checkpoint3 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV{}.h5'.format(FOVSize), verbose=1, save_best_only=True, monitor='val_acc')#'val_acc')
EarlyStop = EarlyStopping(monitor='val_loss', patience=20)
from time import time

#TBLog = TensorBoard(log_dir = '/users/loganjaeger/Desktop/TB/testing_over_ssh/{}'.format(time()))
TBLog = TensorBoard(log_dir = '/home/admin/Desktop/TB/Aug7/{}'.format(round(time(), 4)))
model.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=30,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[Checkpoint1, Checkpoint2, Checkpoint3, Logger, TBLog],
                   class_weight=class_weight
                   )
high_acc = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV{}.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})

def make_and_save_filter_img(layer_number, model = high_acc, pool = None):
  layer_name = 'conv2d_{}'.format(layer_number)
  if pool:
    layer_name = pool
  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[1], (1, 150, 150, 1)))
  s = intermediate_output.shape
  first = int(s[3] / 2)
  sec = int(s[3] - 4)
  plt.subplot(4,3,1)
  plt.imshow(TestNo[1])
  plt.title('original')
  plt.axis('off')
  plt.subplot(4,3,2)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,3)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[10], (1, 150, 150, 1)))
  plt.subplot(4,3,4)
  plt.imshow(TestNo[10])
  plt.axis('off')
  plt.subplot(4,3,5)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,6)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[25], (1, 150, 150, 1)))
  plt.subplot(4,3,7)
  plt.imshow(TestYes[25])
  plt.axis('off')
  plt.subplot(4,3,8)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,9)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[36], (1, 150, 150, 1)))
  plt.subplot(4,3,10)
  plt.imshow(TestYes[36])
  plt.axis('off')
  plt.subplot(4,3,11)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,12)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.savefig('intermediate_output_{}_layer:{}.png'.format(FOVSize, layer_number))
  plt.close()
make_and_save_filter_img(1)
# make_and_save_filter_img(2)
make_and_save_filter_img(3)
# # make_and_save_filter_img(4, pool = "max_pooling2d_2")
# # make_and_save_filter_img(4, pool = "max_pooling2d_3")
make_and_save_filter_img(4)
# # make_and_save_filter_img(5)
# make_and_save_filter_img(6)

DF = h5py.File(os.path.join(DataDir, 'Aug6','side_test.hdf5'), 'r+')
TrainYes_side = DF['TrainYes']
TestYes_side = DF['TestYes']
side_pred_1 = high_acc.predict(np.reshape(TrainYes_side, (len(TrainYes_side), FOVSize, FOVSize, 1)))
side_pred_2 = high_acc.predict(np.reshape(TestYes_side, (len(TestYes_side), FOVSize, FOVSize, 1)))
print('side prediction #1:')
print(len([i for i in side_pred_1 if i > .5]) / len(side_pred_1))
print('side prediction #2:')
print(len([i for i in side_pred_2 if i > .5]) / len(side_pred_2))
print(' ')

DF = h5py.File(os.path.join(DataDir, 'Aug6','new_to_train_500.hdf5'), 'r+')
TestYes_500 = DF['TrainYes']
TestNo_500 = DF['TrainNo']
y_500 = high_acc.predict(np.reshape(TestYes_500, (len(TestYes_500), 500, 500, 1)))
n_500 = high_acc.predict(np.reshape(TestNo_500, (len(TestNo_500), 500, 500, 1)))
print('500x500 w craters:')
print(len([i for i in y_500 if i > .5]) / len(y_500))
print('500x500 no craters:')
print(len([i for i in n_500 if i < .5]) / len(n_500))
print('500x500 total acc:')
print(((len([i for i in y_500 if i > .5]) / len(y_500)) + (len([i for i in n_500 if i < .5]) / len(n_500))) / 2 )
print(' ')

DF = h5py.File(os.path.join(DataDir, 'Aug6','new_to_train_150.hdf5'), 'r+')
TestYes_150 = DF['TrainYes']
TestNo_150 = DF['TrainNo']
y_150 = high_acc.predict(np.reshape(TestYes_150, (len(TestYes_150), 150, 150, 1)))
n_150 = high_acc.predict(np.reshape(TestNo_150, (len(TestNo_150), 150, 150, 1)))
print('150x150 w craters:')
print(len([i for i in y_150 if i > .5]) / len(y_150))
print('150x150 no craters:')
print(len([i for i in n_150 if i < .5]) / len(n_150))
print('150x150 total acc:')
print(((len([i for i in y_150 if i > .5]) / len(y_150)) + (len([i for i in n_150 if i < .5]) / len(n_150))) / 2 )
print(' ')

DF = h5py.File(os.path.join(DataDir, 'Aug6','dif_size_200.hdf5'), 'r+')
TestYes_200 = DF['TestYes']
TestNo_200 = DF['TestNo']
y_200 = high_acc.predict(np.reshape(TestYes_200, (len(TestYes_200), 200, 200, 1)))
n_200 = high_acc.predict(np.reshape(TestNo_200, (len(TestNo_200), 200, 200, 1)))
print('200x200 w craters:')
print(len([i for i in y_200 if i > .5]) / len(y_200))
print('200x200 no craters:')
print(len([i for i in n_200 if i < .5]) / len(n_200))
print(' ')

no_preds = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
vals_y = high_acc.predict(np.reshape(ValYes, (len(ValYes), FOVSize, FOVSize, 1)))
vals_n = high_acc.predict(np.reshape(ValNo, (len(ValNo), FOVSize, FOVSize, 1)))
# plt.subplot(121)
# plt.hist(no_preds, bins = 15)
# plt.title('no craters')
# plt.subplot(122)
# plt.hist(yes_preds, bins = 15)
# plt.title('with craters')
# plt.savefig('CNN_{}_hist.png'.format(FOVSize))
# plt.close()

x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print('no:')
print(x)
print('yes:')
print(y)
print('total:')
print((x + y) / 2)
print('val:')
print((len([i for i in vals_y if i > .5]) + len([i for i in vals_n if i < .5])) / (len(vals_y) + len(vals_n)))
