#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras.backend as K
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

#Specify variables
HomeDir = "/home/admin/Desktop"
DataDir = "RawDataDeploy"
h5_filename = "likely.hdf5"
threshold = .5

#F1 score
def f1_acc(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    if c3 == 0 or c2 == 0:
        return 0
    precision = c1 / c2
    recall = c1 / c3
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

#load the model
model = load_model('/home/admin/Desktop/Saved_CNNs/NFP_actual_acc.h5', custom_objects={'f1_acc': f1_acc})

#load the data
HomeDir = "/home/admin/Desktop"
DataDir = "RawDataDeploy"
h5_filename = "likely_0.hdf5"
datafile_0 = h5py.File(os.path.join(HomeDir, DataDir, h5_filename), 'r')
images_0 = datafile_0["images"]
codes_0 = datafile_0.attrs["codes"]

datafile_1 = h5py.File(os.path.join(HomeDir, DataDir, "likely_two_0.hdf5"), 'r')
images_1 = datafile_1["images"]
codes_1 = datafile_1.attrs["codes"]

#predict
preds_0 = model.predict(images_0)
preds_1 = model.predict(images_1)

#get the codes
def get_codes(predictions, codes, thresh):
	yes_codes = []
	for i, pred in enumerate(predictions):
		if pred > thresh:
			yes_codes.append(codes[i])
	return yes_codes

#write the codes
def write_codes(codes):
	#see what we already have
	already_seen = []
	seen_codes = open("yesCodes.txt", "r")
	for c in seen_codes.read().split_lines():
		already_seen.append(c)
	seen_codes.close()

	verified = []
	verified_txt = open("all_codes_batch1.txt", "r")
	for c in verified_txt.read().split_lines():
		verified.append(c)
	verified_txt.close()

	same_as_v = 0
	same_as_seen = 0
	not_in_either = 0
	
	new_codes = open("1_FI.txt", "w")
	for code in codes:
		if code in verified:
			same_as_seen += 1
			same_as_v += 1
			continue
		elif code in already_seen:
			same_as_seen += 1
			continue
		else:
			not_in_either += 1
			new_codes.write(code)
			new_codes.write("\n")
	new_codes.close()
	print("we got {} / {} of the codes that the last one picked up".format(same_as_seen, len(already_seen)))
	print("we got {} / {} of the codes that I verified from the other one".format(same_as_v, len(verified)))
	print("We found {} new ones".format(not_in_either))



yes_codes_0 = get_codes(preds_0, codes_0, threshold)
yes_codes_1 = get_codes(preds_1, codes_1, threshold)
yes_codes = set()
for i in yes_codes_1:
	yes_codes.add(i)
for j in yes_codes_0:
	yes_codes.add(j)
write_codes(yes_codes)




