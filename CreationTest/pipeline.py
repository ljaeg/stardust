import BlankGather
import CraterCreation
import make_norm
FOVSize = 150
BlankGather.blanks_do(FOVSize)
make_norm.norm_do(FOVSize)
CraterCreation.crater_do(FOVSize)
make_norm.norm_do(FOVSize)
print('!!!!!!!!!!!!!!!!!!!')
print('all done')
print('!!!!!!!!!!!!!!!!!!!')