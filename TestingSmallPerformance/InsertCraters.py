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
RunDir = '/home/admin/Desktop/TestingSmallPerformance'

# try:
#     os.remove(os.path.join(RunDir, 'Data.hdf5'))
# except:
#     pass
# shutil.copy(os.path.join(RunDir, 'Data_10000.hdf5'), os.path.join(RunDir, 'Data.hdf5'))

### LOAD THE HDF.
DataFile = h5py.File(os.path.join(RunDir, 'JustMiddleSmall.hdf5'), 'r+')
TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
NumFOVs = DataFile.attrs['NumFOVs']
Foils = DataFile.attrs['Foils'].split(',')
# Read the Train/Test/Val datasets.
middle = DataFile['middle_small']
side = DataFile['side_small']
blank = DataFile['blanks']


# LOAD CRATER IMAGES AND MAKE AUGMENTED IMAGES
# The augmented images will be scaled, rotated, stretched a bit (aspect ratio).  We will add noise at the input to the CNN, so we don't do that here.
CraterNames = glob(pathname=os.path.join(RunDir, 'Alpha crater images', '*.png'))
Craters = []
for c in CraterNames:
    Craters.append(imread(c)/255)
np.random.seed(42)

def AddCraters(Data, Craters, is_side):
    # We want to randomize the properies of the augmented images.  All the transformation parameters are uniformly distributed except aspect ratio which should hew close to 1 so we use Gaussian.
    scale = np.random.uniform(.08, .3, Data.shape[0])
    rotate = np.random.random(Data.shape[0])*360
    if is_side:
        m1 = np.random.choice([-1, 1], Data.shape[0])
        m2 = np.random.choice([-1, 1], Data.shape[0])
        shift_x = np.random.uniform(.35, .46, (Data.shape[0])) * m1
        shift_y = np.random.uniform(.35, .46, (Data.shape[0])) * m2
        shift = []
        for i in range(Data.shape[0]):
            shift.append([shift_x[i], shift_y[i]])
        shift = np.array(shift)
    else:
        shift = np.random.uniform(-.3, .3, (Data.shape[0], 2))
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

print('Creating side craters.')
AddCraters(side, Craters, True)
print('Creating middle craters.')
AddCraters(middle, Craters, False)

### CLEANUP
DataFile.close()

print('Done.')



# g, a = ImageTools2TransformImage(grayscale, alpha, scale=0.0, rotate=0, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools2TransformImage(grayscale, alpha, scale=0.42, rotate=45, FOVSize=FOVSize)
# imshow(g)
# g, a = ImageTools2TransformImage(grayscale, alpha, scale=1.0, rotate=180, shift=(0.5,0.5), FOVSize=FOVSize)
# plt.imshow(g)
