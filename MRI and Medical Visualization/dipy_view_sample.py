from dipy.data import fetch_sherbrooke_3shell
from os.path import expanduser, join
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table
from dipy.io.image import save_nifti

fetch_sherbrooke_3shell()
home = expanduser('~')
dname = join(home, '.dipy', 'sherbrooke_3shell')
fdwi = join(dname, 'HARDI193.nii.gz')
print(fdwi)
fbval = join(dname, 'HARDI193.bval')
print(fbval)
fbvec = join(dname, 'HARDI193.bvec')
print(fbvec)
data, affine, img = load_nifti(fdwi, return_img=True)
axial_middle = data.shape[2] // 2
plt.figure('Showing the datasets')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(data[:, :, axial_middle, 10].T, cmap='gray', origin='lower')
plt.show()
plt.savefig('data.png', bbox_inches='tight')
gtab = gradient_table(bvals, bvecs)
save_nifti('HARDI193_S0.nii.gz', S0s, affine)
