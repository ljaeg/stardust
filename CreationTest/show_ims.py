import numpy as np 
import matplotlib.pyplot as plt 
import h5py as h5
import os

RD = '/home/admin/Desktop/Aug6'

x = ['A', 'B', 'C', 'D']
y = ['Middle_FOV150_Num10k']
for letter in y:
	dataset = h5.File(os.path.join(RD, '{}.hdf5'.format(letter)), 'r')
	Yes = np.array(dataset['TrainYes'])
	No = np.array(dataset['TrainNo'])
	def do_show(data):
		n = 1
		for im in data:
			plt.imshow(im, cmap = 'gray')
			plt.title('[{}], yes, {}'.format(letter, n))
			n += 1
			plt.axis('off')
			plt.show(block = False)
			plt.waitforbuttonpress(10)
			plt.close()
			if n % 10 == 0:
				x = input('stop and move on? ("y" if so)')
				if x == 'y':
					break

	do_show(Yes)
	do_show(No)


