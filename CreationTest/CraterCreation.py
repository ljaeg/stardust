import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools
import h5py
from scipy.ndimage import measurements as meas
from scipy import ndimage as ndi

### SETUP PARAMETERS
# Raw data is on the Drobo.
RunDir = '/home/admin/Desktop/Aug6'

# try:
#     os.remove(os.path.join(RunDir, 'Data.hdf5'))
# except:
#     pass
# shutil.copy(os.path.join(RunDir, 'Data_10000.hdf5'), os.path.join(RunDir, 'Data.hdf5'))

### LOAD THE HDF.
DataFile = h5py.File(os.path.join(RunDir, 'Middle_FOV150_Num10k_blanks.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
#Foils = DataFile.attrs['Foils'].split(',')
# Read the Train/Test/Val datasets.
TrainNo = DataFile['TrainNo']
TrainYes = DataFile['TrainYes']
TestNo = DataFile['TestNo']
TestYes = DataFile['TestYes']
ValNo = DataFile['ValNo']
ValYes = DataFile['ValYes']

# LOAD CRATER IMAGES AND MAKE AUGMENTED IMAGES
# The augmented images will be scaled, rotated, stretched a bit (aspect ratio).  We will add noise at the input to the CNN, so we don't do that here.
RD = '/home/admin/Desktop/GH'
CraterNames = glob(pathname=os.path.join(RD, 'Alpha_Craters', '*.png'))
Craters = []
for c in CraterNames:
    Craters.append(imread(c)/255)
np.random.seed(42)
print('len of craters: ', len(Craters))

def AddCraters(Data, Craters):
    # We want to randomize the properies of the augmented images.  All the transformation parameters are uniformly distributed except aspect ratio which should hew close to 1 so we use Gaussian.
    scale = np.random.uniform(low = 0, high = 30 / 150, size = Data.shape[0])
    rotate = np.random.random(Data.shape[0])*360
    shift = np.random.uniform(low = .1, high = .9, size = (Data.shape[0],2))-0.5
    aspect = np.random.normal(1, 0.1, Data.shape[0])
    CraterIndices = np.random.randint(len(Craters), size=Data.shape[0])

    # Now make all the transformed craters
    for n, i in enumerate(CraterIndices):
        grayscale = Craters[i][:,:,0]
        alpha = Craters[i][:,:,3]
        g, a = ImageTools.TransformImage(grayscale, alpha, scale=scale[i], rotate=rotate[i], shift=shift[i,:], aspect=aspect[i], FOVSize=FOVSize)
        #print('CraterIndex=%d, scale=%g, rotate=%g, shift=[%g,%g], aspect=%g'%(i, scale[i], rotate[i], shift[i,0], shift[i,1], aspect[i]))

        # Merge the transformed crater with the plain FOV.
        Data[n] = Data[n]*(1-a) + g
        # imshow(Data[n])
        # plt.show()
        if n % 100 == 0:
            print(n)

print('Creating training craters.')
AddCraters(TrainYes, Craters)
print('Creating test craters.')
AddCraters(TestYes, Craters)
print('Creating validation craters.')
AddCraters(ValYes, Craters)

### CLEANUP
DataFile.close()

print('Done.')



# g, a = ImageTools.TransformImage(grayscale, alpha, scale=0.0, rotate=0, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools.TransformImage(grayscale, alpha, scale=0.42, rotate=45, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools.TransformImage(grayscale, alpha, scale=1.0, rotate=180, shift=(0.5,0.5), FOVSize=FOVSize)
# plt.imshow(g)
