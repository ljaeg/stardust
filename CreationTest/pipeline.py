import BlankGather
import CraterCreation
import make_norm
FOVSize = 500
BlankGather.blanks_do(FOVSize)
print('blanks created')
# make_norm.norm_do(FOVSize)
# print('norm number 1')
CraterCreation.crater_do(FOVSize)
print('craters inserted')
make_norm.norm_do(FOVSize)
print('normed')
print('!!!!!!!!!!!!!!!!!!!')
print('all done')
print('!!!!!!!!!!!!!!!!!!!')