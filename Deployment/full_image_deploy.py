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
# h5_filename = "likely.hdf5"
threshold = .9

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
model = load_model('/home/admin/Desktop/Saved_CNNs/NFP_actual_loss.h5', custom_objects={'f1_acc': f1_acc})

#load the data
HomeDir = "/home/admin/Desktop"
DataDir = "RawDataDeploy"
h5_filenames = ["dc_is_zero_{}.hdf5".format(i) for i in list(range(9))]

def predict(hdf_list, mdl):
	preds_all = []
	codes_all = []
	for f in hdf_list:
		datafile = h5py.File(os.path.join(HomeDir, DataDir, f), 'r')
		ims = datafile["images"]
		codes = datafile.attrs["codes"]
		preds = mdl.predict(ims)
		preds_all.extend(preds)
		codes_all.extend(codes)
		datafile.close()
		print(f, " is done")
	more_than33 = [i for i in preds_all if i > .2]
	plt.subplot(111)
	plt.hist(more_than33)
	plt.savefig("histogramOfPreds384512.png")
	print("the fraction of the predictions greater than .2 is {}".format(len(more_than33) / len(preds_all)))
	return preds_all, codes_all

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
	for c in seen_codes.read().splitlines():
		already_seen.append(c)
	seen_codes.close()

	verified = []
	verified_txt = open("all_codes_batch1.txt", "r")
	for c in verified_txt.read().splitlines():
		verified.append(c)
	verified_txt.close()

	same_as_v = 0
	same_as_seen = 0
	not_in_either = 0
	
	new_codes = open("1_FI.txt", "w")
	for code in codes:
		code = code.decode('UTF-8')
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



ps, cs = predict(h5_filenames, model)
yes_codes = get_codes(ps, cs, threshold)
write_codes(yes_codes)




