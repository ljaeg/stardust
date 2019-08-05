import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools2
import h5py
from scipy.ndimage import measurements as meas
from scipy import ndimage as ndi

### SETUP PARAMETERS
# Raw data is on the Drobo.
RunDir = '/home/admin/Desktop/Preprocess'

# try:
#     os.remove(os.path.join(RunDir, 'Data.hdf5'))
# except:
#     pass
# shutil.copy(os.path.join(RunDir, 'Data_10000.hdf5'), os.path.join(RunDir, 'Data.hdf5'))

### LOAD THE HDF.
DataFile = h5py.File(os.path.join(RunDir, 'FOV150_Num10000_noside.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
Foils = DataFile.attrs['Foils']
# Read the Train/Test/Val datasets.
TrainYes = DataFile['TrainYes']
TrainNo = DataFile['TrainNo']
TestYes = DataFile['TestYes']
TestNo = DataFile['TestNo']
ValYes = DataFile['ValYes']
ValNo = DataFile['ValNo']

# LOAD CRATER IMAGES AND MAKE AUGMENTED IMAGES
# The augmented images will be scaled, rotated, stretched a bit (aspect ratio).  We will add noise at the input to the CNN, so we don't do that here.
CraterNames = glob(pathname=os.path.join('/home/admin/Desktop/GH', 'Alpha_Craters', '*.png'))
Craters = []
for c in CraterNames:
    Craters.append(imread(c)/255)
np.random.seed(42)

def AddCraters(Data, Craters):
    # We want to randomize the properies of the augmented images.  All the transformation parameters are uniformly distributed except aspect ratio which should hew close to 1 so we use Gaussian.
    scale = np.random.uniform(low = 0, high = .2, size = Data.shape[0])
    rotate = np.random.random(Data.shape[0])*360
    shift = np.random.uniform(low = .1, high = .9, shape = (Data.shape[0], 2)) - .5
    aspect = np.random.normal(1, 0.1, Data.shape[0])
    CraterIndices = np.random.randint(len(Craters), size=Data.shape[0])

    # Now make all the transformed craters
    for n, i in enumerate(CraterIndices):
        grayscale = Craters[i][:,:,0]
        alpha = Craters[i][:,:,3]
        g, a = ImageTools2.TransformImage(grayscale, alpha, scale=scale[i], rotate=rotate[i], shift=shift[i,:], aspect=aspect[i], FOVSize=FOVSize)
        #print('CraterIndex=%d, scale=%g, rotate=%g, shift=[%g,%g], aspect=%g'%(i, scale[i], rotate[i], shift[i,0], shift[i,1], aspect[i]))

        # Merge the transformed crater with the plain FOV.
        Data[n] = Data[n]*(1-a) + g
        # imshow(Data[n])
        # plt.show()
        if n % 100 == 0:
            print(n)


from time import time
start_time = time()
print('Creating TRAINING craters.')
AddCraters(TrainYes, Craters)
elapsed = time() - start_time
print('It has taken {} minutes up to this point. We have about {} minutes left'.format(elapsed / 60, (2 * elapsed) / 60))

print('Creating VALIDATION craters.')
AddCraters(ValYes, Craters)
elapsed = time() - start_time
print('It has taken {} minutes up to this point. We have about {} minutes left'.format(elapsed / 60, (elapsed / 2) / 60))

print('Creating TESTING craters')
AddCraters(TestYes, Craters)
elapsed = time() - start_time
print('It took {} minutes, my golly that was a trek'.format(elapsed / 60))

### CLEANUP
DataFile.close()

print('Done.')



# g, a = ImageTools2TransformImage(grayscale, alpha, scale=0.0, rotate=0, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools2TransformImage(grayscale, alpha, scale=0.42, rotate=45, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools2TransformImage(grayscale, alpha, scale=1.0, rotate=180, shift=(0.5,0.5), FOVSize=FOVSize)
# plt.imshow(g)
