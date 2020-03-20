"""
Fully connected pipeline. Specify FOVSize and NumImgs, and this will make a hdf5 file with the datasets
TrainYes, TrainNo, TestYes, TestNo, ValYes, and ValNo, each with the number of images equal to NumImgs,
and the image size FOVSize x FOVSize x 1 (grayscale).
The datasets TrainYes, TestYes, and ValYes will all contain craters.
All of the datasets will be normalized in one of two ways (or both ways), which can be changed in the make_norm.py file.
The hdf5 file will be saved under the path SavePath.
"""

import BlankGather
import CraterCreation
import make_norm
import os

FOVSize = 30
NumImgs = 10
Dir = "/home/admin/Desktop/Aug6"
Name = 'test_Mar20.hdf5'
SavePath = os.path.join(Dir, Name)

### Gather blank backgrounds
BlankGather.blanks_do(FOVSize, NumImgs, SavePath)
print('blanks created')

### Insert craters into appropriate sets
CraterCreation.crater_do(FOVSize, SavePath)
print('craters inserted')

### Normalize the images
make_norm.norm_do(FOVSize)
print('normed')

print('!!!!!!!!!!!!!!!!!!!')
print('all done')
print('!!!!!!!!!!!!!!!!!!!')
