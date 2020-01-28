#This is a deployment method biased against false positives. As Zack says, instant science
#This will HOPEFULLY give me a few false positives and a couple of true positives

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle


#load the models for the different sizes
model150 = load_model('/home/admin/Desktop/Saved_CNNs/NFP_acc_FOV150.h5')
model100 = load_model('/home/admin/Desktop/Saved_CNNs/NFP_loss_FOV100.h5', custom_objects={'f1_acc': f1_acc})
model30 = load_model("/home/admin/Desktop/GH/NFP30_F1.h5", custom_objects={'f1_acc': f1_acc})

#specify the thresholds for the different sizes
th_150 = .5
th_100 = .5
th_30 = .5

NumImages = 10000 #number of images to look through

#We need to normalize the image in some way
def norm1(im):
	#just put between 0 and 1
	mn = np.min(im)
	mx = np.max(im)
	return (im - mn) / (mx - mn)

def norm2(im):
	#just mean subtraction
	mean = np.mean(im)
	return im - mean 

def norm3(im):
	#mean subtraction and std of 1
	mean = np.mean(im)
	std = np.std(im)
	return (im - mean) / std 

def split_predict_150(im):
	a = [0, 100, 200, 234]
	b = [0, 100, 200, 300, 362]
	lower_preds = [0]
	for i in a:
		w = i + 150
		for j in b:
			z = j + 150
			sub_img = (image[i:w, j:z]).reshape(1, 150, 150, 1)
			sin = norm1(sub_img)
			pred = model.predict(sin)
			if pred > th_150:
				new_pred = split_predict_100(sin)
				lower_preds.append(new_pred)
	return max(lower_preds)

def split_predict_100(im):
	#takes a 150x150 img and predicts on it using 100x100 imgs
	a = [0, 50]
	pred_30s = [0]
	for i in a:
		w = i + 100
		for j in a:
			z = j + 100
			sub_img = im[i:w, j:z].reshape(1, 100, 100, 1)
			sin = norm1(sub_img)
			pred = model100.predict(sin)
			if pred > th_100:
				pred30 = split_predict_30(sin)
				pred_30s.append(pred30)
	return max(pred_30s)

def split_predict_30(im):
	a = [0, 15, 30, 45, 60, 70]
	preds = []
	for i in a:
		w = i + 30
		for j in a:
			z = j + 30
			sub_img = im[i:w, j:z].reshape(1, 30, 30, 1)
			sin = norm1(sub_img)
			pred = model30.predict(sin)
			if pred > th_30:
				#early cutoff
				return 1
	return 0

def is_control(im):
	s = im.shape
	if s[0] == 385 or s[0] == 513:
		return 1
	else:
		return 0
	

# for i in range(NumImages):
# 	#load image
# 	#get image ID, maybe URL
# 	img = ...
# 	if is_control(im):
# 		#do the same thing but add to a control group
# 		#go to next iteration
# 	pred = split_predict_150(img)
# 	if pred == 1:
# 		#add url to txt file
# 		"""
# 		then I'll have a program that goes thru the txt file,
# 		shows me, a human, the image, and I'll determine if it 
# 		has a crater or not. Boo ya
# 		"""

def find_codes(ims, codes):
	yes_codes = []
	i = 0
	for im in ims:
		x = split_predict_150(im)
		if x == 1:
			yes_codes.append(codes[i])
		i += 1
	print("We got {} positives here".format(len(yes_codes)))
	for code in yes_codes:
		print(code)
	print(" ")
	return yes_codes


for i in range(12):
	name = "20181207_" + str(i)
	f = h5py.File("/home/admin/Desktop/RawDataDeploy/" + name + ".hdf5")
	codes = f.attrs["codes"]
	ims = f["images"]
	find_codes(ims, codes)
















