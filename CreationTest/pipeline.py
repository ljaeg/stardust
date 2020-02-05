import BlankGather
import CraterCreation
import make_norm
FOVSize = 30

BlankGather.blanks_do(FOVSize)
print('blanks created')

CraterCreation.crater_do(FOVSize)
print('craters inserted')

make_norm.norm_do_large(FOVSize)
print('normed')

print('!!!!!!!!!!!!!!!!!!!')
print('all done')
print('!!!!!!!!!!!!!!!!!!!')