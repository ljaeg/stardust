import h5py as h5
import numpy as np 
from glob2 import glob
import os
from imageio import imread
#we're going to norm the craters before we put them in, hopefully it helps with the realistic-ness

RunDir = '/home/admin/Desktop/GH'

CraterNames = glob(pathname=os.path.join(RunDir, 'Alpha_Craters', '*.png'))
Craters = []
for c in CraterNames:
    Craters.append(imread(c) / 255)

for i in Craters:
	print(np.mean(i))
	print(np.max(i))
	print(np.min(i))
	print(' ')


def norm(dataset):
	#We want to ignore all of the parts of the image that have been edited out
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

