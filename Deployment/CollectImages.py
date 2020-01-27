#This is a program for going thru the txt file and collecting images of craters 
#To store on the machine at Berkeley (Big Dusty).

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests

Dir = "/home/admin/Desktop/RawDataDeploy/"
fname = "20181207.txt"

"""
I'd like to store the images in an hdf5 file for more compact storage.
The metadata will be stored on a different datasite in the same hdf5 file
"""

def get_image(code):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	#r = urllib.request.urlopen(url)
	r = requests.get(url)
	img = Image.open(BytesIO(r.content))
	img = np.array(img) / 255.0
	return np.reshape(img, (384, 512, 1))

def get_img_array(fname):
	#(N, 384, 512, 1) image array
	path = Dir + fname
	ims = []
	codes = []
	with open(path) as f:
		for line in f:
			code = str(line)
			codes.append(code)
			im = get_image(code)
			ims.append(im)
	ims = np.array(ims)
	codes = np.array(codes)
	return arr, codes

def make_dataset(dataset_name, directory, codes_fname):
	ims, codes = get_img_array(codes_fname)
	datafile = h5py.File(directory + dataset_name + ".hdf5", "w")
	image_set = datafile.create_dataset("images", ims.shape, data = ims)
	codes_set = datafile.create_dataset("codes", codes.shape, data = codes)
	datafile.close()


make_dataset("20181207", Dir, fname)

