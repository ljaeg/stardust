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

FOVSize = 200

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

high_acc = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_extra.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})
high_acc.summary()

def calc_test_acc(name):
  DF = h5py.File(os.path.join(DataDir, 'Aug6','{}.hdf5'.format(name)), 'r+')
  TestYes = DF['TestYes']
  TestNo = DF['TestNo']
  FOVSize = DF.attrs['FOVSize']
  y = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
  n = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
  cp = len([i for i in y if i > .5]) / len(y)
  cn = len([i for i in n if i < .5]) / len(n)
  n = np.round(n, 3)
  print(FOVSize,' w craters:')
  print(cp)
  print(FOVSize,' no craters:')
  print(cn)
  print(FOVSize, ' total acc:')
  print((cp + cn) / 2)
  print(' ')
  return (cp, cn)

cp, cn = calc_test_acc('new_to_train_500')

def Bayes(P_cp, P_fp, P_crater):
  return (P_cp * P_crater) / ((P_cp * P_crater) + (P_fp * (1 - P_crater)))
b = Bayes(cp, (1 - cn), 1 / 100000)
print('probability that a flagged image actually has a crater:')
print(b)
print(' ')
E = 1/b
V = np.sqrt((1-b)/(b**2))
print('expected value of craters flagged until first crater actually found: ')
print(E)
#print(V)


