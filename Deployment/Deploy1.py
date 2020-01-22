#This is for giving me the X number of images that are most likely to have a crater
#Note that this is the most naive way of doing this. For loops for everything

"""
Considerations:
1. load the NxN model
2. split up images into W NxN subimages (remember to include overlap)
3. For each image, predict on all W subimages
4. For each image, record the highest score a single subimage recieved, call this the overall score
5. Read out the X images with the highest overall score
"""

#import all the things
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np 
import tensorflow as tf 
import keras.backend as K
import h5py 
from keras.models import Sequential, load_model, Model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


FOVSize = 200 #Size of each subimage is FOVSize x FOVSize
"""
model = load_model() #The model we want to use to predict on the subimages
overlap = 50 #the overlap
full_ims = h5py.File("placeholder", "r+") #or however we read in the actual files
"""
X = 100 #the number of images we'll return

def split_image_and_pred(image, sub_size, overlap):
	a = 0
	b = sub_size
	all_ps = []
	while b < image.shape[0]:
		c = 0
		d = sub_size
		while d < image.shape[1]:
			x = image[a:b, c:d]
			p = model.predict(x)
			all_ps.append(p)
			c += sub_size - overlap 
			d += sub_size - overlap 
		a += sub_size - overlap 
		b += sub_size - overlap 
	return max(all_ps)



def find_top(images):
	#we need to return some id associated with each image
	#For now, we assume the images are numbered 0 through X-1
	i = 0
	d = {}
	for image in images:
		pred = split_image_and_pred(image, FOVSize, overlap)
		d[i] = pred 
	vals = d.values().sort()
	lastval = vals[49]
	return [k for k in d.keys() if d[k] >= lastval]


def just_split(image, sub_size, overlap):
	a = 0
	b = sub_size
	i=1
	while b < image.shape[0]:
		c = 0
		d = sub_size
		while d < image.shape[1]:
			x = image[a:b, c:d]
			if i <= 9:
				plt.subplot(3, 3, i)
				plt.imshow(x)
				plt.axis("off")
				i+=1
			c += sub_size - overlap 
			d += sub_size - overlap
		a += sub_size - overlap 
		b += sub_size - overlap 
	plt.show()
	

img = mpimg.imread("/users/loganjaeger/Downloads/fc_7H4Mtu2eC83o7mHxnbpA.21333-001.jpg")
print(img.shape)
just_split(img, 100, 25)



