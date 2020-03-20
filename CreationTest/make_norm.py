import h5py as h5
import numpy as np 
import os
import matplotlib

#This performs mean subtraction and divides each image by its standard devation, so each image is centered at zero and has a std of 1
def mean_subtract(dataset):
	s = dataset.shape #get the shape 
	m = np.mean(dataset, axis = (1, 2)) #get the mean of every image
	m = np.repeat(m, np.repeat(s[2]*s[1], s[0])) #broadcast it into a usable shape
	m = np.reshape(m, s)

	std = np.std(dataset, axis = (1, 2)) #get standard deviation of each image
		for n, i in enumerate(std):
			if i == 0:
				std[n] = 1 #Necessary so we don't divide by zero
	std = np.repeat(std, np.repeat(s[2]*s[1], s[0])) #broadcast into a usable shape
	std = np.reshape(std, s)

	new = (dataset - m) / std #normalize data
	return new

#This simply puts the range of values in the image to 0-1 (a simpler way that would be nearly equal is to just do: dataset / 255, but many have different ranges than 0-255)
def zero_to_one(dataset):
	s = dataset.shape
	max_vals = np.max(dataset, axis = (1, 2)) #get maximums
	min_vals = np.min(dataset, axis = (1, 2)) #get minimums

	#make sure we don't divide by 0
	bools = max_vals == min_vals
	np.place(max_vals, bools, 1)
	np.place(min_vals, bools, 0)

	#broadcast into usable shapes
	max_vals = np.repeat(max_vals, np.repeat(s[2] * s[1], s[0]))
	max_vals = np.reshape(max_vals, s)

	min_vals = np.repeat(min_vals, np.repeat(s[2] * s[1], s[0]))
	min_vals = np.reshape()

	new = (dataset - min_vals) / (max_vals - min_vals) #norm
	return new

#Do both normalization methods.
def both(dataset):
	new = mean_subtract(dataset)
	new = zero_to_one(new)
	return new

#default to both
def determine_normtype(st):
	if st == "mean_subtract":
		return mean_subtract
	elif st == "zero_to_one":
		return zero_to_one
	else:
		return both


def norm_do(size, UnNormedPath, NormedPath = "whatever", norm_type = "both"):
	# This one saves the normed files to a different file, so you can compare performance or 
	Dir = '/home/admin/Desktop/Aug6'
	not_normed = h5.File(UnNormedPath, 'r+') #load in the images

	FOVSize = not_normed.attrs['FOVSize']
	NumFOVs = not_normed.attrs['NumFOVs']
	try:
		Foils = not_normed.attrs['Foils'].split(',')
	except:
		Foils = not_normed.attrs['Foils']
	print(Foils)

	# Read the Train/Test/Val datasets.
	TrainNo = np.array(not_normed['TrainNo'])
	TrainYes = np.array(not_normed['TrainYes'])
	TestNo = np.array(not_normed['TestNo'])
	TestYes = np.array(not_normed['TestYes'])
	ValNo = np.array(not_normed['ValNo'])
	ValYes = np.array(not_normed['ValYes'])
	
	norm = determine_normtype(norm_type)

	not_normed["TrainNo"] = norm(TrainNo)

	not_normed.close()


	# normed = h5.File(NormedPath, 'w') #By writing over the same name, we destroy the unnormed file 
	# normed.attrs.create('FOVSize', FOVSize)
	# normed.attrs.create("NumFOVs", NumFOVs)
	# normed.attrs.create('Foils', Foils)

	# new_TrainYes = norm(TrainYes)
	# normed.create_dataset('TrainYes', shape = new_TrainYes.shape, dtype = new_TrainYes.dtype, data = new_TrainYes)

	# new_TrainNo = norm(TrainNo)
	# normed.create_dataset("TrainNo", shape = new_TrainNo.shape, dtype = new_TrainNo.dtype, data = new_TrainNo)

	# new_TestYes = norm(TestYes)
	# normed.create_dataset('TestYes', shape = new_TestYes.shape, dtype = new_TrainNo.dtype, data= new_TestYes)

	# new_TestNo = norm(TestNo)
	# normed.create_dataset('TestNo', shape = new_TestNo.shape, dtype = new_TestNo.dtype, data = new_TestNo)

	# new_ValYes = norm(ValYes)
	# normed.create_dataset('ValYes', shape = new_ValYes.shape, dtype = new_ValYes.dtype, data = new_ValYes)

	# new_ValNo = norm(ValNo)
	# normed.create_dataset("ValNo", shape = new_ValNo.shape, dtype = new_ValNo.dtype, data = new_ValNo)

	# normed.close()

def norm_do_large(size, PathToFile):
	# Currently, we write over the 
	
	not_normed = h5.File(PathToFile, 'r+')

	FOVSize = not_normed.attrs['FOVSize']
	NumFOVs = not_normed.attrs['NumFOVs']
	try:
		Foils = not_normed.attrs['Foils'].split(',')
	except:
		Foils = not_normed.attrs['Foils']
	print(Foils)
	# Read the Train/Test/Val datasets.
	TrainNo = not_normed['TrainNo']
	TrainYes = not_normed['TrainYes']
	TestNo = not_normed['TestNo']
	TestYes = not_normed['TestYes']
	ValNo = not_normed['ValNo']
	ValYes = not_normed['ValYes']

	def norm(dataset):
		for n, im in enumerate(dataset):
			m = np.mean(im)
			s = np.std(im)
			if not(s):
				s = 1
			new = (im - m) / s
			dataset[n] =  new
			if not(n % 50):
				print(n)
	norm(TrainYes)
	norm(TrainNo)
	norm(TestYes)
	norm(TestNo)
	norm(ValYes)
	norm(ValNo)
	not_normed.close()
