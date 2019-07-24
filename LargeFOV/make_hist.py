import numpy as np 
import matplotlib.pyplot as plt 
import os
import h5py 

DataDir = '/home/admin/Desktop/Preprocess'
DataFile = h5py.File(os.path.join(DataDir, 'FOV100_Num10000_normed_w_std.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
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

a = np.ndarray.flatten(np.array(TrainNo))
b = np.ndarray.flatten(np.array(TrainYes))
plt.subplot(141)
plt.hist(a, bins = 15)
plt.hist(b, bins = 15)

print(np.mean(TrainYes[0]))

plt.subplot(142)
plt.hist(np.ndarray.flatten(TrainNo[0, :, :]), bins = 10)
plt.hist(np.ndarray.flatten(TrainNo[10, :, :]), bins = 10)
plt.hist(np.ndarray.flatten(TrainNo[60, :, :]), bins = 10)

plt.subplot(143)
plt.hist(np.ndarray.flatten(TrainYes[0, :, :]), bins = 10)
plt.hist(np.ndarray.flatten(TrainYes[11, :, :]), bins = 10)
plt.hist(np.ndarray.flatten(TrainYes[41, :, :]), bins = 10)

plt.subplot(144)
plt.imshow(TrainYes[0])
plt.colorbar()
plt.show()
