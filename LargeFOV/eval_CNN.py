import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import keras.backend as K
import h5py 
from keras.models import Sequential, load_model, Model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

DataDir = '/home/admin/Desktop'
# DataFile = h5py.File(os.path.join(DataDir, 'FOV150_Num10000_normed_01.hdf5'), 'r+')
# #TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
# FOVSize = DataFile.attrs['FOVSize']
# NumFOVs = DataFile.attrs['NumFOVs']
# try:
#   Foils = DataFile.attrs['Foils'].split(',')
# except:
#   Foils = DataFile.attrs['Foils']
# # Read the Train/Test/Val datasets.
# num_ims = int(NumFOVs*.33)
# ad_sub = 0
# TrainNo = np.array(DataFile['TrainNo'])[:num_ims]
# TrainYes = np.array(DataFile['TrainYes'])[:num_ims]
# TestNo = np.array(DataFile['TestNo'])[:num_ims]
# TestYes = np.array(DataFile['TestYes'])[:num_ims]
# ValNo = np.array(DataFile['ValNo'])[:num_ims]
# ValYes = np.array(DataFile['ValYes'])[:num_ims]

FOVSize = 150

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

high_acc = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV{}.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})

def calc_test_acc(name):
  DF = h5py.File(os.path.join(DataDir, 'Aug6','{}.hdf5'.format(name)), 'r+')
  TestYes = DF['TestYes']
  TestNo = DF['TestNo']
  FOVSize = DF.attrs['FOVSize']
  y = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
  n = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
  cp = len([i for i in y if i > .5]) / len(y)
  cn = len([i for i in n if i < .5]) / len(n)
  print(FOVSize,' w craters:')
  print(cp)
  print(FOVSize,' no craters:')
  print(cn)
  print(FOVSize, ' total acc:')
  print((cp + cn) / 2)
  print(' ')

calc_test_acc('new_to_train_150')

"""
calc_test_acc('new_to_train_500')
calc_test_acc('new_to_train_200')

DF1 = h5py.File(os.path.join(DataDir, 'Aug6','new_to_train_500.hdf5'), 'r+')
#These are 500x500 pixel data
TrainYes = DF1['TrainYes']
TrainNo = DF1['TrainNo']
ValYes = DF1['ValYes']
ValNo = DF1['ValNo']

TrainData = np.concatenate((TrainNo,TrainYes), axis=0)[:,:,:,np.newaxis]
ValData = np.concatenate((ValNo,ValYes), axis=0)[:,:,:,np.newaxis]


# And make answer vectors
TrainAnswers = np.ones(len(TrainNo) + len(TrainYes))
TrainAnswers[:len(TrainNo)] = 0
ValAnswers = np.ones(len(ValNo) + len(ValYes))
ValAnswers[:len(ValNo)] = 0

# Make generators to stream them.
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
batch_size = 4
train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)

Checkpoint1 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_F1_extra.h5', verbose=1, save_best_only=True, monitor='val_f1_acc')#'val_acc')
Checkpoint2 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_loss_extra.h5', verbose=1, save_best_only=True, monitor='val_loss')#'val_acc')
Checkpoint3 = ModelCheckpoint('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_extra.h5', verbose=1, save_best_only=True, monitor='val_acc')#'val_acc')
EarlyStop = EarlyStopping(monitor='val_loss', patience=20)
from time import time

#TBLog = TensorBoard(log_dir = '/users/loganjaeger/Desktop/TB/testing_over_ssh/{}'.format(time()))
TBLog = TensorBoard(log_dir = '/home/admin/Desktop/TB/Aug7/{}/second_train_w_200'.format(round(time(), 4)))
high_acc.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=35,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[TBLog, Checkpoint1, Checkpoint2, Checkpoint3],
                   initial_epoch = 25
                   )

x = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_extra.h5', custom_objects={'f1_acc': f1_acc})

y_500 = x.predict(np.reshape(TestYes_500, (len(TestYes_500), 500, 500, 1)))
n_500 = x.predict(np.reshape(TestNo_500, (len(TestNo_500), 500, 500, 1)))
after_y = len([i for i in y_500 if i > .5]) / len(y_500)
after_n = len([i for i in n_500 if i < .5]) / len(n_500)
print('500x500 w craters:')
print(after_y)
print('500x500 no craters:')
print(after_n)
print(' ')
print('change for yes:')
change1 = round((after_y - b4_y) * 100, 4)
print(change1, '%')
print('change for no:')
change2 = round((after_n - b4_n) * 100, 4)
print(change2, '%')

def Bayes(P_cp, P_fp, P_crater):
    return (P_cp * P_crater) / ((P_cp * P_crater) + (P_fp * (1 - P_crater)))
print(Bayes(after_y, (1 - after_n), 1 / 100000))

# model.summary()

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
#   plt.savefig('intermediate_output_{}_layer{}.png'.format(FOVSize, layer_number))
#   plt.close()
# make_and_save_filter_img(1)
# make_and_save_filter_img(2)
# make_and_save_filter_img(3)
# make_and_save_filter_img(4)
# make_and_save_filter_img(5)
# make_and_save_filter_img(6)

DF = h5py.File(os.path.join(DataDir, 'Aug6','Middle_FOV150_Num10k_new.hdf5'), 'r+')
TestNo = DF['TestNo']
TestYes = DF['TestYes']


no_preds = model.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = model.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
#vals_y = model.predict(np.reshape(ValYes, (len(ValYes), FOVSize, FOVSize, 1)))
#vals_n = model.predict(np.reshape(ValNo, (len(ValNo), FOVSize, FOVSize, 1)))
plt.subplot(121)
plt.hist(no_preds, bins = 15)
plt.title('no craters')
plt.subplot(122)
plt.hist(yes_preds, bins = 15)
plt.title('with craters')
plt.savefig('CNN_{}_hist.png'.format(FOVSize))

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
"""