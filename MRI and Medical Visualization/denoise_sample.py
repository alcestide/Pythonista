
import numpy as np
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti


t1_fname = get_fnames('stanford_t1')
data, affine = load_nifti(t1_fname)
mask = data > 1500
print("vol size", data.shape)

sigma = estimate_sigma(data, N=32)
t = time()
den = nlmeans(data, sigma=sigma, mask=mask, patch_radius=1,
              block_radius=2, rician=True)

print("total time", time() - t)
axial_middle = data.shape[2] // 2
before = data[:, :, axial_middle].T
after = den[:, :, axial_middle].T
difference = np.abs(after.astype(np.float64) - before.astype(np.float64))
difference[~mask[:, :, axial_middle].T] = 0


fig, ax = plt.subplots(1, 3)
ax[0].imshow(before, cmap='gray', origin='lower')
ax[0].set_title('before')
ax[1].imshow(after, cmap='gray', origin='lower')
ax[1].set_title('after')
ax[2].imshow(difference, cmap='gray', origin='lower')
ax[2].set_title('difference')
plt.show()
plt.savefig('denoised.png', bbox_inches='tight')
save_nifti('denoised.nii.gz', den, affine)
