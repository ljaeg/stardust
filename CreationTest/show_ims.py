import numpy as np 
import matplotlib.pyplot as plt 
import h5py as h5
import os

RD = '/home/admin/Desktop/Aug6'

x = ['A', 'B', 'C', 'D']
y = ['Middle_FOV150_Num10k']
# for letter in y:
# 	dataset = h5.File(os.path.join(RD, '{}.hdf5'.format(letter)), 'r')
# 	Yes = np.array(dataset['TrainYes'])
# 	No = np.array(dataset['TrainNo'])
# 	def do_show(data):
# 		n = 1
# 		for im in data:
# 			plt.imshow(im, cmap = 'gray')
# 			plt.title('[{}], yes, {}'.format(letter, n))
# 			n += 1
# 			plt.axis('off')
# 			plt.show(block = False)
# 			plt.waitforbuttonpress(10)
# 			plt.close()
# 			if n % 10 == 0:
# 				x = input('stop and move on? ("y" if so)')
# 				if x == 'y':
# 					break

# 	do_show(Yes)
# 	do_show(No)
A = 'Middle_FOV150_Num10k'
B = 'Middle_FOV150_Num10k_new'
nb = h5.File(os.path.join(RD, '{}.hdf5'.format(A)), 'r')
b = h5.File(os.path.join(RD, '{}.hdf5'.format(B)), 'r')

ValY_nb = np.array(nb['ValYes'])
TestY_nb = np.array(nb['TestYes'])

ValY_b = np.array(b['TrainYes'])
TestY_b = np.array(b['TestYes'])

plt.subplot(231)
plt.imshow(ValY_nb[0], cmap = 'gray')
plt.axis('off')
plt.title('nb 0 V')

plt.subplot(232)
plt.imshow(ValY_nb[2500], cmap = 'gray')
plt.axis('off')
plt.title('nb 2500 V')

plt.subplot(233)
plt.imshow(TestY_nb[2600])
plt.axis('off')
plt.title('nb 2600 T')

plt.subplot(234)
plt.imshow(ValY_b[0], cmap = 'gray')
plt.axis('off')
plt.title('b 0 V')

plt.subplot(235)
plt.imshow(ValY_b[2500], cmap = 'gray')
plt.axis('off')
plt.title('b 2500 V')

plt.subplot(236)
plt.imshow(TestY_b[2600])
plt.axis('off')
plt.title('b 2600 T')

plt.savefig('ims.png')