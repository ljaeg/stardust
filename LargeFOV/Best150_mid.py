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
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization, SpatialDropout2D, AveragePooling2D
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

# Train/validate/test info
batch_size=int(512/16)
class_weight={0: 1, 1: 1}
epochs = 100
ConvScale=32 / 2
DenseScale=64
# GN1 = .054
# GN2 = .018
# GN3 = .14
# alpha = .24
spatial_d_rate = 0.25
GN1 = 0
GN2 = 0
GN3 = 0
alpha = 0
dropout_rate = 0.3
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
DataFile = h5py.File(os.path.join(DataDir, 'Aug6','Middle_FOV150_Num10k_new.hdf5'), 'r+')
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

y1 = TrainYes[200]
y2 = TrainYes[305]
n1 = TrainNo[200]
n2 = TrainNo[305]


plt.subplot(321)
plt.hist(np.ndarray.flatten(y1))
plt.title('y1')
plt.subplot(322)
plt.hist(np.ndarray.flatten(y2))
plt.title('y2')
plt.subplot(323)
plt.hist(np.ndarray.flatten(n1))
plt.title('n1')
plt.subplot(324)
plt.hist(np.ndarray.flatten(n2))
plt.title('n2')
plt.subplot(325)
plt.hist(np.ndarray.flatten(np.array(TrainYes)))
plt.title('all yes')
plt.subplot(326)
plt.hist(np.ndarray.flatten(np.array(TrainNo)))
plt.title('all no')
plt.savefig('no_side.png')


# TrainNo = TrainNo[:num_ims]
# print('after:', len(TrainNo))
# TrainYes = TrainYes[:num_ims]
# TestNo = TestNo[:num_ims]
# TestYes = TestYes[:num_ims]
# ValNo = ValNo[:num_ims]
# ValYes = ValYes[:num_ims]


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
#input_shape = (FOVSize, FOVSize, 1) # Only one channel since these are B&W.
input_shape = (FOVSize, FOVSize, 1)

model = Sequential()
model.add(GaussianNoise(GN1, input_shape = input_shape))
model.add(Conv2D(int(4*ConvScale), (kernel_size, kernel_size), padding='valid', input_shape=input_shape, kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
# model.add(GaussianNoise(GN2))
model.add(Conv2D(int(4*ConvScale), (kernel_size, kernel_size), padding='valid', input_shape=input_shape, kernel_regularizer = regularizers.l2(reg_scale)))
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

model.add(Flatten())
model.add(Dense(int(4*DenseScale)))
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
                   epochs=20,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[Checkpoint1, Checkpoint2, Checkpoint3, Logger, TBLog],
                   class_weight=class_weight
                   )
high_acc = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV{}.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})

# def make_and_save_filter_img(layer_number, pool = None):
#   layer_name = 'conv2d_{}'.format(layer_number)
#   if pool:
#     layer_name = pool
#   intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#   intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[1], (1, 150, 150, 1)))
#   s = intermediate_output.shape
#   first = int(s[3] / 2)
#   sec = int(s[3] - 1)
#   plt.subplot(4,3,1)
#   plt.imshow(TestNo[1])
#   plt.title('original')
#   plt.axis('off')
#   plt.subplot(4,3,2)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')
#   plt.subplot(4,3,3)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')

#   intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[10], (1, 150, 150, 1)))
#   plt.subplot(4,3,4)
#   plt.imshow(TestNo[10])
#   plt.axis('off')
#   plt.subplot(4,3,5)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')
#   plt.subplot(4,3,6)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')

#   intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[25], (1, 150, 150, 1)))
#   plt.subplot(4,3,7)
#   plt.imshow(TestYes[25])
#   plt.axis('off')
#   plt.subplot(4,3,8)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')
#   plt.subplot(4,3,9)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')

#   intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[36], (1, 150, 150, 1)))
#   plt.subplot(4,3,10)
#   plt.imshow(TestYes[36])
#   plt.axis('off')
#   plt.subplot(4,3,11)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')
#   plt.subplot(4,3,12)
#   plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
#   plt.axis('off')
#   plt.savefig('intermediate_output_{}_layer:{}.png'.format(FOVSize, layer_number))
# make_and_save_filter_img(1)
# make_and_save_filter_img(2)
# make_and_save_filter_img(3)
# make_and_save_filter_img(4, pool = "max_pooling2d_2")
# make_and_save_filter_img(4, pool = "max_pooling2d_3")
# # make_and_save_filter_img(4)
# # make_and_save_filter_img(5)
# # make_and_save_filter_img(6)

DF = h5py.File(os.path.join(DataDir, 'Aug6','Middle_FOV150_Num10k_new.hdf5'), 'r+')
TestNo = DF['TestNo']
TestYes = DF['TestYes']


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


# testing_data = h5py.File('/home/admin/Desktop/ForGit/TestingSmallPerformance/JustMiddleSmall.hdf5')
# middles = testing_data['middle_small']
# sides = testing_data['side_small']
# blanks = testing_data['blanks']

# #HERE I am standardizing this data. I know for certain this isn't standardized
# middles = standardize_exp(middles)
# sides = standardize_exp(sides)
# blanks = standardize_exp(blanks)

# middles = np.reshape(middles, (990, 30, 30, 1))
# sides = np.reshape(sides, (990, 30, 30, 1))
# blanks = np.reshape(blanks, (990, 30, 30, 1))

# middle_pred = model.predict(middles)
# side_pred = model.predict(sides)
# blank_pred = model.predict(blanks)

# middle_wrong = [i for i in middle_pred if i < .5]
# side_wrong = [i for i in side_pred if i < .5]
# blank_wrong = [i for i in blank_pred if i > .5]

# mid_acc = 1 - (len(middle_wrong) / len(middle_pred))
# side_acc = 1 - (len(side_wrong) / len(side_pred))
# blank_acc = 1 - (len(blank_wrong) / len(blank_pred))

# print('middle accuracy: ', mid_acc)
# print('side accuracy: ', side_acc)
# print('blank accuracy: ', blank_acc)
# print('overall accuracy: ', (mid_acc+side_acc+blank_acc) / 3)

# import matplotlib.pyplot as plt
# plt.subplot(231)
# plt.hist(np.array(middle_pred))
# plt.title('middle ({})'.format(round(mid_acc, 2)))

# plt.subplot(232)
# plt.hist(np.array(side_pred))
# plt.title('side ({})'.format(round(side_acc, 2)))

# plt.subplot(233)
# plt.hist(np.array(blank_pred))
# plt.title('blank ({})'.format(round(blank_acc, 2)))

# plt.subplot(234)
# plt.hist(np.array(middle_wrong))
# plt.title('middle wrong')

# plt.subplot(235)
# plt.hist(np.array(side_wrong))
# plt.title('side wrong')

# plt.subplot(236)
# plt.hist(np.array(blank_wrong))
# plt.title('blank wrong')

# plt.savefig('MiddleVsSide.png')


# predicted = model.predict(np.reshape(TestData, (3300 * 2, 30, 30, 1)))
# no_tags = []
# yes_tags = []
# no = []
# yes = []
# file = h5py.File('/home/admin/Desktop/wrong_classifications.hdf5', 'w')
# # for n, r in enumerate(predicted):
# #   if n < 3300:
# #     #im = TestNo[n]
# #     #no.append(im)
# #     no_tags.append(r)
# #   else:
# #     #im = TestYes[n - 3300]
# #     #yes.append(im)
# #     yes_tags.append(r)

# # no = np.array(no)
# # yes = np.array(yes)

# no_tags = predicted[:3300]
# yes_tags = predicted[3300:]


# # false_pos = file.create_dataset('false_pos', shape = no.shape, data = no)
# # false_neg = file.create_dataset('false_neg', shape = yes.shape, data = yes)


# # print('####')
# # print('yes: ', len(yes))
# # print('no: ', len(no))
# # print('total wrong: ', len(yes) + len(no))
# # print('percent wrong: ', (len(no) + len(yes)) / (6600))
# # print('percent of the wrong that are false negatives: ', (len(yes) / (len(no) + len(yes))))

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2)
# axs[0].hist(np.array(yes_tags), bins = 10)
# axs[1].hist(np.array(no_tags), bins = 10)
# plt.title('+ // -')
# plt.savefig('answer_space.png')

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
