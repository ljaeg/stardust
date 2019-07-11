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

from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from scipy.misc import imread

# Train/validate/test info
batch_size=int(512 * 1)
class_weight={0: 1, 1: 1}
epochs = 250
ConvScale=2 
DenseScale=1 
GN1 = .03
GN2 = .05
GN3 = 0

# Calculate the F1 score which we use for optimizing the CNN.
def f1_acc(y_true, y_pred):

    # import numpy as np

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
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

# Load the image datasets from the HDF.
# RunDir = '/home/zack/Data/SAH/Code/Gen002/001 - CNN'
# DataDir = '/home/zack/Data/SAH/Code/Gen002/Data'
DataDir = '/home/admin/Desktop/Preprocess'
DataFile = h5py.File(os.path.join(DataDir, 'Data_10000_craters.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
Foils = DataFile.attrs['Foils'].split(',')
# Read the Train/Test/Val datasets.
TrainNo = DataFile['TrainNo']
TrainYes = DataFile['TrainYes']
TestNo = DataFile['TestNo']
TestYes = DataFile['TestYes']
ValNo = DataFile['ValNo']
ValYes = DataFile['ValYes']

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
model.add(GaussianNoise(GN1, input_shape = input_shape))
model.add(Conv2D(int(32*ConvScale), (3, 3), padding='valid', input_shape=input_shape))
model.add(LeakyReLU(alpha = 0))
model.add(GaussianNoise(GN2))
model.add(Conv2D(int(32*ConvScale), (3,3), padding='valid', input_shape=input_shape))
model.add(LeakyReLU(alpha = 0))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(int(64*ConvScale), (3,3), padding='valid'))
model.add(LeakyReLU(alpha = 0))
model.add(GaussianNoise(GN3))
model.add(Conv2D(int(64*ConvScale), (3,3), padding='valid'))
model.add(LeakyReLU(alpha = 0))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(int(512*DenseScale)))
model.add(LeakyReLU(alpha = 0))
model.add(Dropout(0.5))

model.add(Dense(int(128*DenseScale)))
model.add(LeakyReLU(alpha = 0))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Nadam(lr=0.0002), loss='binary_crossentropy', metrics=['acc', f1_acc])
model.save('Foils_CNN.h5')
model = load_model('Foils_CNN.h5', custom_objects={'f1_acc': f1_acc})
model.summary()
# plot_model(model, to_file='Foils_CNN.png', show_shapes=True)



# Do the training
# CSVLogger is a checkpoint function.  After each epoch, it will write the stats from that epoch to a csv file.
Logger = CSVLogger('Foils_CNN_Log.txt', append=True)
# ModelCheckpoint will save the configuration of the network after each epoch.
# save_best_only ensures that when the validation score is no longer improving, we don't overwrite
# the network with a new configuration that is overfitting.
Checkpoint1 = ModelCheckpoint('Foils_CNN_F1.h5', verbose=1, save_best_only=True, monitor='val_f1_acc')#'val_acc')
Checkpoint2 = ModelCheckpoint('Foils_CNN_loss.h5', verbose=1, save_best_only=True, monitor='val_loss')#'val_acc')
Checkpoint3 = ModelCheckpoint('Foils_CNN_acc.h5', verbose=1, save_best_only=True, monitor='val_acc')#'val_acc')
EarlyStop = EarlyStopping(monitor='val_loss', patience=20)
from time import time
TBLog = TensorBoard(log_dir = '/home/admin/Desktop/TB/Zack_CNN/my_data/zack_data/July9/batchsize={}/convscale={}/DenseScale={}/GN_at_start={}/GN2={}/GN3={}/Lrelu_test_shouldbeGS'.format(batch_size, ConvScale, DenseScale, GN1, GN2, GN3))

model.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=175,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[Checkpoint1, Checkpoint2, Checkpoint3, Logger, TBLog],
                   class_weight=class_weight
                   )

predicted = model.predict(np.reshape(TestData, (3300 * 2, 30, 30, 1)))
no_tags = []
yes_tags = []
no = []
yes = []
file = h5py.File('/home/admin/Desktop/wrong_classifications.hdf5', 'w')
# for n, r in enumerate(predicted):
#   if n < 3300:
#     #im = TestNo[n]
#     #no.append(im)
#     no_tags.append(r)
#   else:
#     #im = TestYes[n - 3300]
#     #yes.append(im)
#     yes_tags.append(r)

# no = np.array(no)
# yes = np.array(yes)

no_tags = predicted[:3300]
yes_tags = predicted[3300:]


# false_pos = file.create_dataset('false_pos', shape = no.shape, data = no)
# false_neg = file.create_dataset('false_neg', shape = yes.shape, data = yes)


# print('####')
# print('yes: ', len(yes))
# print('no: ', len(no))
# print('total wrong: ', len(yes) + len(no))
# print('percent wrong: ', (len(no) + len(yes)) / (6600))
# print('percent of the wrong that are false negatives: ', (len(yes) / (len(no) + len(yes))))

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2)
axs[0].hist(np.array(yes_tags), bins = 10)
axs[1].hist(np.array(no_tags), bins = 10)
plt.show()

# Plot the learning curve.
# logresult = pd.read_csv('Foils_CNN_Log.txt', delimiter=',', index_col='epoch')
# logresult.reset_index(inplace=True)
# ax1 = logresult.plot(ylim=(0.995,1))
# ax2 = logresult.plot(ylim=(0,0.005))
# ax1.get_figure().savefig('LearningCurve-Accuracy.png')
# ax2.get_figure().savefig('LearningCurve-Loss.png')


#
# # Now run it against the actual full data and see how well it works.
# # Grab a subimage making sure that it stays within the original image.
# SubImageStride = 5 # Subimages will have their centroids in a grid with this spacing.
#
#
# def MakeBounds(x, y, imgshape):
#     # Remember x and y for the image are y and x as reported by the coordinate file.
#     mmin, mmax = y - dx, y + dx
#     nmin, nmax = x - dx, x + dx
#
#     # If this x,y coordinate would select a square (radius dx) outside the
#     # image, then simply push the square to stay within the image.
#     if y < dx:
#         mmin, mmax = 0, dx * 2
#     if x < dx:
#         nmin, nmax = 0, dx * 2
#     if x > imgshape[1] - dx:
#         nmin, nmax = imgshape[1] - 2 * dx, imgshape[1] + 1
#     if y > imgshape[0] - dx:
#         mmin, mmax = imgshape[0] - 2 * dx, imgshape[0] + 1
#
#     return mmin, mmax, nmin, nmax
#
# def ChopImage(img):
#     SubImages = []
#     SubImageCentroids = []
#
#     # Make a grid of points which are the centroids for all the sub images.
#     xvals = range(dx,
#                   img.shape[1]-dx+SubImageStride, # Go as far to the edge as possible -- there may be a little extra overlap on a couple images.
#                   SubImageStride)
#     yvals = range(dx,
#                   img.shape[0]-dx+SubImageStride,
#                   SubImageStride)
#
#     # Extract all those images.
#     for y in yvals:
#         for x in xvals:
#             mmin, mmax, nmin, nmax = MakeBounds(x,y, img.shape)
#             SubImages.append(img[mmin:mmax, nmin:nmax])
#             SubImageCentroids.append((x,y))
#
#     # Turn the list of subimages into an input for the CNN.
#     SubImages = np.array(SubImages).reshape(len(SubImages),dx*2,dx*2,1)
#     SubImageCentroids = np.array(SubImageCentroids).reshape(len(SubImageCentroids), 2)
#
#     return(SubImages, SubImageCentroids)
#
# def FindCratersInImage(ImageName):
#     img = imread(ImageName, flatten=True)
#     SubImages, SubImageCentroids = ChopImage(img)
#     CNN_prediction = model.predict(SubImages)
#     Craters = np.where(CNN_prediction > 0.99)
#     if(len(Craters[0]) == 0):
#         # No craters
#         x_avg = None
#         y_avg = None
#     else:
#         # Yes crater(s)
#         x_avg = np.median(SubImageCentroids[Craters[0],0])
#         y_avg = np.median(SubImageCentroids[Craters[0],1])
#
#     return x_avg, y_avg, len(Craters[0])
#
# def HowManyCratersInDir(DirName):
#     Images = os.listdir(DirName)
#     np.random.seed(5)
#     Images = shuffle(Images, random_state=5)
#     CraterImages = 0
#     SumImages = 0
#     for Img in Images[:200]:
#         if Img == '.DS_Store':
#             continue
#         xpos,ypos,NumHits = FindCratersInImage(os.path.join(DirName, Img))
#         if xpos is not None:
#             CraterImages += 1
#         print(NumHits, end='_')
#         SumImages +=1
#     print()
#     print(CraterImages, ' out of ', SumImages, ' have craters.')
#
#
# model = load_model('Foils_CNN_F1.h5', custom_objects={'f1_acc': f1_acc})
# print('Compute Stats based on F1 score')
# print('No Craters:')
# HowManyCratersInDir(os.path.join(RawDataDir,'NoCraters'))
# print('\nWith Craters:')
# HowManyCratersInDir(os.path.join(RawDataDir,'WithCraters'))
