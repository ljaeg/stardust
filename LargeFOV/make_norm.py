import h5py as h5
import numpy as np 
import os

##We're going to norm the images by per image mean subtraction, and save them to a DIFFERENT directory so that we compare performance
Dir = '/home/admin/Desktop/Preprocess'
not_normed = h5.File(os.path.join(Dir, 'FOV100_Num10000_b.hdf5'), 'r')
# testdir = '/users/loganjaeger/Desktop/SAH/Code/Current'
# not_normed = h5.File(os.path.join(testdir, 'Data_1000_craters.hdf5'), 'r')

FOVSize = not_normed.attrs['FOVSize']
NumFOVs = not_normed.attrs['NumFOVs']
Foils = not_normed.attrs['Foils'].split(',')
# Read the Train/Test/Val datasets.
TrainNo = not_normed['TrainNo']
TrainYes = not_normed['TrainYes']
TestNo = not_normed['TestNo']
TestYes = not_normed['TestYes']
ValNo = not_normed['ValNo']
ValYes = not_normed['ValYes']

normed = h5.File(os.path.join(Dir, 'FOV100_Num10000_normed_w_std_and_redvar.hdf5'), 'w')
normed.attrs.create('FOVSize', FOVSize)
normed.attrs.create("NumFOVs", NumFOVs)
normed.attrs.create('Foils', Foils)

def norm(dataset):
	print(np.mean(dataset))

	m = np.mean(dataset, axis = (1, 2))
	s = dataset.shape
	m = np.repeat(m, np.repeat(s[2]*s[1], s[0]))
	m = np.reshape(m, s)

	std = np.std(dataset, axis = (1, 2))
	for n, i in enumerate(std):
		if i == 0:
			std[n] = 1
	std = np.repeat(std, np.repeat(s[2]*s[1], s[0]))
	std = np.reshape(std, s)

	new = (dataset - m) / std

	p = np.min(new, axis = (1, 2))
	q = np.max(new, axis = (1, 2))
	for i in range(len(p)):
		if q[i] - p[i] == 0:
			q[i] = 1
			p[i] = 0
	p = np.repeat(p, np.repeat(s[2]*s[1], s[0]))
	q = np.repeat(q, np.repeat(s[2]*s[1], s[0]))
	p = np.reshape(p, s)
	q = np.reshape(q, s)

	new = (new - p) / (q - p)
	#new = new - .5
	return new

new_TrainYes = norm(TrainYes)
normed.create_dataset('TrainYes', shape = new_TrainYes.shape, dtype = new_TrainYes.dtype, data = new_TrainYes)

new_TrainNo = norm(TrainNo)
normed.create_dataset("TrainNo", shape = new_TrainNo.shape, dtype = new_TrainNo.dtype, data = new_TrainNo)

new_TestYes = norm(TestYes)
normed.create_dataset('TestYes', shape = new_TestYes.shape, dtype = new_TrainNo.dtype, data= new_TestYes)

new_TestNo = norm(TestNo)
normed.create_dataset('TestNo', shape = new_TestNo.shape, dtype = new_TestNo.dtype, data = new_TestNo)

new_ValYes = norm(ValYes)
normed.create_dataset('ValYes', shape = new_ValYes.shape, dtype = new_ValYes.dtype, data = new_ValYes)

new_ValNo = norm(ValNo)
normed.create_dataset("ValNo", shape = new_ValNo.shape, dtype = new_ValNo.dtype, data = new_ValNo)

normed.close()