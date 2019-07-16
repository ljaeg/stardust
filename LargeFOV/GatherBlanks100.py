import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools2
import h5py

### SETUP PARAMETERS
# Raw data is on the Drobo.
RawDir = '/home/admin/Desktop/Preprocess'
Foils = ['I1009N', 'I1126N', 'I1126N_2', 'I1126N_3']
FOVSize = 100 # 30 pixels squadwddddare for each image.
NumFOVs = 10000 # How many FOVs to extract from the raw data.
TrainTestValSplit = [0.33, 0.33, 0.33]
NumTrain = int(NumFOVs*TrainTestValSplit[0])
NumTest = int(NumFOVs*TrainTestValSplit[1])
NumVal = int(NumFOVs*TrainTestValSplit[2])

### SCAN THE RAW DATA
# We don't need to redo globbing if we already globbed.
try:
    with open('/home/admin/Desktop/Preprocess/GlobbedFilesPC_New.txt', 'r') as f:
        print('here')
        GlobbedFiles = f.read().splitlines()
except IOError as e:
    print('there')
    GlobbedFiles = []
    for d in Foils:
        d = os.path.join(RawDir, d, '*.tif')
        g = glob(pathname=d)
        GlobbedFiles += list(g)
    with open('GlobbedFilesPC_New.txt', 'w') as f:
        f.writelines('%s\n' % n for n in GlobbedFiles)
print('There are %d image files in the raw data.' % len(GlobbedFiles))

### MAKE HDF TO HOLD OUR IMAGES.
DataFile = h5py.File('/home/admin/Desktop/Preprocess/FOV100_Num10000_b.hdf5', 'w')
DataFile.attrs['TrainTestValSplit'] = TrainTestValSplit
DataFile.attrs['FOVSize'] = FOVSize
DataFile.attrs['NumFOVs'] = NumFOVs
DataFile.attrs['Foils'] = ', '.join(Foils)

# Yes means this image has a crater.
# No means there is no crater.
compression_opts = 2
TrainYes = DataFile.create_dataset('TrainYes', (NumTrain, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)
TrainNo  = DataFile.create_dataset('TrainNo',  (NumTrain, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)

TestYes  = DataFile.create_dataset('TestYes', (NumTest, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)
TestNo   = DataFile.create_dataset('TestNo',  (NumTest, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)

ValYes   = DataFile.create_dataset('ValYes', (NumVal, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)
ValNo    = DataFile.create_dataset('ValNo',  (NumVal, FOVSize, FOVSize), dtype='f8', chunks=True, compression='gzip', compression_opts=compression_opts)

DataFile.flush()

### POPULATE TRAIN/TEST/VAL WITH NO CRATER IMAGES
# Choose random files from which to draw FOVs.
np.random.seed(1121)

def GetRandomFOVs(GlobbedFiles, Data, SpeedupFactor=20):
    # It is slow to read each image.  So we will take SpeedupFactor FOVs from each in order to speed up I/O.
    NumFOVs = Data.shape[0] # The number of images to get is the first axis of the data cube the caller wants filled.
    FOVSize = Data.shape[1] # x for each image.
    assert(Data.shape[1] == Data.shape[2]) # x == y for each image.

    Files = np.random.choice(GlobbedFiles, int(NumFOVs/SpeedupFactor)+1) # We need +1 in case SpeedupFactor is not a divisor.
    for i, n in enumerate(Files):
        img = imread(n)
        for j in range(SpeedupFactor):
            # Stop when we have filled our quota.
            if (i*SpeedupFactor+j) >= NumFOVs:
                break
            # Pull a FOV out at random and put it in the HDF.
            FOV = ImageTools2.GetRandomFOV(img, FOVSize)
            Data[i*SpeedupFactor+j,:,:] = FOV/255.0
        print('%s: #%d'%(Data.name, (i*SpeedupFactor+j)))

# Grab all the FOVs we want.
from time import time
starting_time = time()

GetRandomFOVs(GlobbedFiles, TrainYes)
DataFile.flush()
elapsed = time() - starting_time
print('it has taken {} seconds. It will take {} more seconds, or {} more minutes!'.format(elapsed, elapsed * 5, (elapsed * 5) / 60))

GetRandomFOVs(GlobbedFiles, TrainNo)
DataFile.flush()
elapsed = time() - starting_time
print('it has taken {} seconds. It will take {} more seconds, or {} more minutes!'.format(elapsed, (elapsed *2), (elapsed * 2)/ 60))

GetRandomFOVs(GlobbedFiles, TestYes)
DataFile.flush()
elapsed = time() - starting_time
print('it has taken {} seconds. It will take {} more seconds, or {} more minutes!'.format(elapsed, elapsed, elapsed / 60))

GetRandomFOVs(GlobbedFiles, TestNo)
DataFile.flush()
elapsed = time() - starting_time
print('it has taken {} seconds. It will take {} more seconds, or {} more minutes!'.format(elapsed, elapsed / 2, (elapsed / 2) / 60))

GetRandomFOVs(GlobbedFiles, ValNo)
DataFile.flush()
elapsed = time() - starting_time
print('it has taken {} seconds. It will take {} more seconds, or {} more minutes!'.format(elapsed, elapsed / 5, (elapsed / 5) / 60))

GetRandomFOVs(GlobbedFiles, ValYes)
DataFile.flush()
elapsed = time() - starting_time
print("Whoo! We're done, that took {} minutes".format(round(elapsed / 60, 3)))

### CLEANUP
DataFile.close()
