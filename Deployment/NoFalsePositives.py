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

#f1 score
def f1_acc(y_true, y_pred):
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
    return f1_score

#load the models for the different sizes
model150 = load_model('/home/admin/Desktop/Saved_CNNs/NFP_acc_FOV150.h5', custom_objects={'f1_acc': f1_acc})
model100 = load_model('/home/admin/Desktop/Saved_CNNs/NFP_loss_FOV100.h5', custom_objects={'f1_acc': f1_acc})
model30 = load_model("/home/admin/Desktop/GH/NFP30_F1.h5", custom_objects={'f1_acc': f1_acc})

#specify the thresholds for the different sizes
th_150 = .5
th_100 = .55
th_30 = .8

NumImages = 10000 #number of images to look through

#We need to normalize the image in some way
def norm1(im):
	#just put between 0 and 1
	mn = np.min(im)
	mx = np.max(im)
	if mx == mn:
		return im - mn
	return (im - mn) / (mx - mn)

def norm2(im):
	#just mean subtraction
	mean = np.mean(im)
	return im - mean 

def norm3(im):
	#mean subtraction and std of 1
	mean = np.mean(im)
	std = np.std(im)
	if std == 0:
		return im - mean
	return (im - mean) / std 

def norm_all(im):
	temp = norm3(im)
	return norm1(temp)

def norm_most(im):
	temp = norm2(im)
	return norm1(im)

def split_predict_150(im):
	a = [0, 100, 200, 234]
	b = [0, 100, 200, 300, 362]
	lower_preds = [0]
	for i in a:
		w = i + 150
		for j in b:
			z = j + 150
			sub_img = (im[i:w, j:z]).reshape(1, 150, 150, 1)
			sin = norm_all(sub_img)
			pred = model150.predict(sin)
			if pred > th_150:
				new_pred = split_predict_30(sin)
				lower_preds.append(new_pred)
	return max(lower_preds)

def split_predict_100(im):
	#takes a 150x150 img and predicts on it using 100x100 imgs
	#print("here")
	im = im.reshape(150, 150)
	a = [0, 50]
	pred_30s = [0]
	for i in a:
		w = i + 100
		for j in a:
			z = j + 100
			# print(im.shape)
			# print(im[i:w, j:z].shape)
			# print(i)
			# print(w)
			# print(j)
			# print(z)
			# print(' ')
			sub_img = (im[i:w, j:z]).reshape(1, 100, 100, 1)
			#sin = norm_all(sub_img)
			sin = sub_img
			# x = split_predict_30(sin)
			pred_30s.append(x)
			pred = model100.predict(sin)
			if pred > th_100:
				pred30 = split_predict_30(sin)
				pred_30s.append(pred30)
	return max(pred_30s)

def split_predict_30(im):
	#print("there")
	a = [0, 15, 30, 45, 60, 70]
	b = [0, 20, 40, 60, 80, 100, 120]
	a = b
	im = im.reshape(150, 150)
	preds = []
	for i in a:
		w = i + 30
		for j in a:
			z = j + 30
			sub_img = (im[i:w, j:z]).reshape(1, 30, 30, 1)
			#sin = norm_all(sub_img)
			sin = sub_img
			pred = model30.predict(sin)
			if pred > th_30:
				#early cutoff
				#plt.imsave(str(pred[0][0]) + ".png", np.reshape(sin, (30, 30)), cmap = "gray")
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

def actual():
	file = open("yesCodes.txt", "w")
	for i in ["likely_two_0", "likely_0"]:
		#name = "20181207_" + str(i)
		name = i
		f = h5py.File("/home/admin/Desktop/RawDataDeploy/" + name + ".hdf5")
		codes = f.attrs["codes"]
		ims = f["images"]
		codes = find_codes(ims, codes)
		for code in codes:
			file.write(code.decode('UTF-8'))
			file.write("\n")



#Testing below:
def testing_positives(ims):
	pos = 0
	neg = 0
	i = 0
	for im in ims:
		x = split_predict_150(im)
		if x == 1:
			pos += 1
		else:
			neg += 1
		i += 1
		if (i % 150) == 0:
			print(i)
	acc = pos / (pos + neg)
	print("accuracy on pos: ", acc)
	print(' ')
	return acc

def testing_negatives(ims):
	pos = 0
	neg = 0
	i = 0
	for im in ims:
		x = split_predict_150(im)
		if x == 1:
			pos += 1
		else:
			neg += 1
		i += 1
		if (i % 150) == 0:
			print(i)
	acc = neg / (pos + neg)
	print("accuracy on neg: ", acc)
	print(' ')
	return acc

def testing():
	ps = []
	ns = []
	for name in ["testing_0"]:
		f = h5py.File("/home/admin/Desktop/RawDataDeploy/" + name + ".hdf5")
		ims = f["images"]
		acc_p = testing_positives(ims)
		ps.append(acc_p)
	for name in ["negatives_0", "negatives_1", "negatives_2"]:
		f = h5py.File("/home/admin/Desktop/RawDataDeploy/" + name + ".hdf5")
		ims = f["images"]
		acc_n = testing_negatives(ims)
		ns.append(acc_n)
	print("positive accuracy: ", np.mean(ps))
	print("negative accuracy: ", np.mean(ns))


testing()






