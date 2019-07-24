import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import os
from keras.models import load_model
import keras.backend as K

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def f1_acc(y_true, y_pred):

    # import numpy as np

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
    # if K.isnan(f1_score):
    #     print('c1:', c1)
    #     print('c2:', c2)
    #     print('c3:', c3)

    return f1_score

DataDir = '/home/admin/Desktop/Preprocess'
DataFile = h5py.File(os.path.join(DataDir, 'FOV100_Num10000_normed_w_std.hdf5'), 'r+')
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
try:
  Foils = DataFile.attrs['Foils'].split(',')
except:
  Foils = DataFile.attrs['Foils']


# Read the Train/Test/Val datasets.
TrainNo = DataFile['TrainNo']
TrainYes = DataFile['TrainYes']
TestNo = DataFile['TestNo']
TestYes = DataFile['TestYes']
ValNo = DataFile['ValNo']
ValYes = DataFile['ValYes']

TestData = np.concatenate((TestNo,TestYes), axis=0)[:,:,:,np.newaxis]
TestAnswers = np.ones(len(TestNo) + len(TestYes))
TestAnswers[:len(TestNo)] = 0
inds = np.random.randint(low = 0, high = len(TestAnswers), size = 100)

ims = np.array([TestData[i] for i in inds])
s = ims.shape
ims = np.reshape(ims, (s[0], s[1], s[2], 1))
ans = [TestAnswers[i] for i in inds]
model = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV100.h5', custom_objects={'f1_acc': f1_acc})
preds = model.predict(ims)

total_right = 0
network_wrong = 0
n = 0
for i in inds:
	plt.imshow(np.reshape(TestData[i], (100, 100)), cmap = 'gray')
	plt.show(block = False)
	plt.waitforbuttonpress(10)
	plt.close()
	human_answer = input('Y or N: ')
	assert human_answer == 'y' or human_answer == 'n' or human_answer == 'Y' or human_answer == 'N', 'please enter valid answer'

	if human_answer == 'y' or human_answer == 'Y':
		x = 1
	elif human_answer == 'n' or human_answer == 'N':
		x = 0
	if x == TestAnswers[i]:
		print('SUCCESS!!!!!!')
		total_right += 1
	else:
		print('failure :(')
	n += 1
	print('{}/{}'.format(n, len(inds)))
	print('index: {}'.format(i))
	print('our model gave this a score of {}'.format(preds[n - 1]))
	if preds[n -1] > .5 and TestAnswers[i] == 0:
		print('THE NETWORK GOT THIS ONE WRONG')
		network_wrong += 1
	if preds[n - 1] < .5 and TestAnswers[i] == 1:
		print('THE NETWORK GOT THIS ONE WRONG')
		network_wrong += 1




print('###########')
print('FINAL SCORE: ')
print('you got {} right out of {} images!'.format(total_right, len(inds)))
print('the network got {} wrong'.format(network_wrong))
print('###########')
