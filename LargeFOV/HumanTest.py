import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import os

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

TrainData = np.concatenate((TrainNo,TrainYes), axis=0)[:,:,:,np.newaxis]
TrainAnswers = np.ones(len(TrainNo) + len(TrainYes))
TrainAnswers[:len(TrainNo)] = 0
inds = np.random.randint(low = 0, high = len(TrainAnswers), size = 100)

g = input('type something: ')
print(g)