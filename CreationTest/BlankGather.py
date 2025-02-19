import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools
import h5py

def blanks_do(FOVSize, NumFOVs, PathToFile):
    ### SETUP PARAMETERS
    # Raw data is stored locally.
    RawDir = '/home/admin/Desktop/NEW_Images'
    Foils = ['I1009N', 'I1126N', 'I1126N_2', 'I1126N_3']
    TrainTestValSplit = [.33, .33, .33] #currently this hace no bearing on anything except the attribute in the hdf5 file
    
    NumTrain = NumFOVs
    NumTest = NumFOVs
    NumVal = NumFOVs
     
    ### SCAN THE RAW DATA
    # We don't need to redo globbing if we already globbed.
    try:
        with open('GlobbedFiles_Aug6.txt', 'r') as f:
            GlobbedFiles = f.read().splitlines()
    except IOError as e:
        GlobbedFiles = []
        for d in Foils:
            d = os.path.join(RawDir, d, '*.tif')
            g = glob(pathname=d)
            GlobbedFiles += list(g)
        with open('GlobbedFiles_Aug6.txt', 'w') as f:
            f.writelines('%s\n' % n for n in GlobbedFiles)
    print('There are %d image files in the raw data.' % len(GlobbedFiles))

    ### MAKE HDF TO HOLD OUR IMAGES.
    DataFile = h5py.File(PathToFile, 'w')
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
    np.random.seed(5)

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
                FOV = ImageTools.GetRandomFOV(img, FOVSize)
                Data[i*SpeedupFactor+j,:,:] = FOV/255.0
            print('%s: #%d'%(Data.name, (i*SpeedupFactor+j)))

    # Grab all the FOVs we want.
    GetRandomFOVs(GlobbedFiles, TrainNo)
    DataFile.flush()
    GetRandomFOVs(GlobbedFiles, TrainYes)
    DataFile.flush()
    GetRandomFOVs(GlobbedFiles, TestNo)
    DataFile.flush()
    GetRandomFOVs(GlobbedFiles, TestYes)
    DataFile.flush()
    GetRandomFOVs(GlobbedFiles, ValNo)
    DataFile.flush()
    GetRandomFOVs(GlobbedFiles, ValYes)
    DataFile.flush()

    ### CLEANUP
    DataFile.close()
