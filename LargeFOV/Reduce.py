import numpy as np 
import matplotlib.pyplot as plt 
import os 
import skimage
import h5py 

DataDir = '/home/admin/Desktop/Aug6'
DataFile = h5py.File(os.path.join(DataDir,'new_to_train_500.hdf5'), 'r+')

TrainYes = DataFile['TrainYes']
TrainNo = DataFile['TrainNo']

def make_plt(x, title, im):
	plt.subplot(x)
	plt.axis('off')
	plt.title(title)
	plt.imshow(im, cmap = 'gray')

def view():
	c1 = TrainYes[31]
	# c2 = TrainYes[99]
	# nc1 = TrainNo[20]
	# nc2 = TrainNo[99]

	c1_r = skimage.measure.block_reduce(c1, (20, 20), func = np.max)
	# c2_r = skimage.measure.block_reduce(c2, (3, 3), func = np.mean)
	# nc1_r = skimage.measure.block_reduce(nc1, (3, 3), func = np.mean)
	# nc2_r = skimage.measure.block_reduce(nc2, (3, 3), func = np.mean)

	make_plt(121, 'original', c1)
	make_plt(122, 'reduced (np.max)', c1_r)
	# make_plt(423, 'c2 orig', c2)
	# make_plt(424, 'c2 red', c2_r)
	# make_plt(425, 'nc1 orig', nc1)
	# make_plt(426, 'nc1 red', nc1_r)
	# make_plt(427, 'nc2 orig', nc2)
	# make_plt(428, 'nc2 red', nc2_r)
	plt.show()

def reduce_whole_ds(ds, block_size = (2, 2), f = np.max):
	new_ds = []
	for im in ds:
		new_im = skimage.measure.block_reduce(im, block_size, func = f)
		new_ds.append(new_im)
	return np.array(new_ds)

