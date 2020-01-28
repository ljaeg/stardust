#This is a program for going thru the txt file and collecting images of craters 
#To store on the machine at Berkeley (Big Dusty).

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import time

Dir = "/home/admin/Desktop/RawDataDeploy/"
fname = "likely_2.txt"
save_f_base = "likely_two"

Save_to_Dir = Dir
step_size = 623
steps = 1
start_number = 0

"""
I'd like to store the images in an hdf5 file for more compact storage.
The metadata will be stored on a different dataset in the same hdf5 file
"""

def get_image(code):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	#r = urllib.request.urlopen(url)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
	except OSError:
		print("got error from URL")
		img = np.ones((384, 512, 1))
	return np.reshape(img, (384, 512, 1))

def get_img_array(fname, start, step_size):
	#(N, 384, 512, 1) image array
	path = Dir + fname
	ims = []
	codes = []
	with open(path) as f:
		for line in f.read().splitlines()[start:start+step_size]:
			code = str(line)
			codes.append(code)
			im = get_image(code)
			ims.append(im)
	ims = np.array(ims, dtype = "f8")
	return ims, codes

def make_dataset(dataset_name, save_dir, codes_fname, start, step_size):
	ims, codes = get_img_array(codes_fname, start, step_size)
	datafile = h5py.File(save_dir + dataset_name + ".hdf5", "w")
	image_set = datafile.create_dataset("images", ims.shape, data = ims)
	#codes_set = datafile.create_dataset("codes", codes.shape, data = codes, dtype = "U29")
	datafile.attrs["codes"] = np.string_(codes)
	datafile.close()


def do_incrimentally(step_size, start, number_of_steps, save_dir, save_f_base, codes_fname):
	print("you are doing a total of {} images in {} parts".format(step_size*number_of_steps, number_of_steps))
	print("I predict a total time of something like {} minutes".format(predict_total_t(step_size * number_of_steps)))
	s = start 
	step = 0
	t = time.time()
	while step < number_of_steps:
		ds_name = save_f_base + "_" + str(step)
		make_dataset(ds_name, save_dir, codes_fname, s, step_size)
		print("saved " + ds_name)
		s += step_size
		step += 1
		time_til_done(step_size*number_of_steps, step*step_size, time.time() - t)

def predict_total_t(total):
	return (total * 11.5) / (50 * 60)

def time_til_done(total_N, current_N, current_time):
	total_t = (total_N * current_time) / current_N
	remaining_t = total_t - current_time
	print("seconds left: ", remaining_t)
	print("minutes left: ", remaining_t / 60)
	print("hours left: ", remaining_t / (60*60))
	print(" ")

do_incrimentally(step_size, start_number, steps, Dir, save_f_base, fname)
print("DONE!")





















