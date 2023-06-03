#============================================
#SNR estimation for Diffusion-Weighted Images
#============================================

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

print('Computing brain mask...')
b0_mask, mask = median_otsu(data, vol_idx=[0])

print('Computing tensors...')
tenmodel = TensorModel(gtab)
tensorfit = tenmodel.fit(data, mask=mask)

print('Computing worst-case/best-case SNR using the corpus callosum...')
from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

threshold = (0.6, 1, 0, 0.1, 0, 0.1)
CC_box = np.zeros_like(data[..., 0])

mins, maxs = bounding_box(mask)
mins = np.array(mins)
maxs = np.array(maxs)
diff = (maxs - mins) // 4
bounds_min = mins + diff
bounds_max = maxs - diff

CC_box[bounds_min[0]:bounds_max[0],
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1

mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold,
                                     return_cfa=True)

save_nifti('cfa_CC_part.nii.gz', (cfa*255).astype(np.uint8), affine)
save_nifti('mask_CC_part.nii.gz', mask_cc_part.astype(np.uint8), affine)

import matplotlib.pyplot as plt
region = 40
fig = plt.figure('Corpus callosum segmentation')
plt.subplot(1, 2, 1)
plt.title("Corpus callosum (CC)")
plt.axis('off')
red = cfa[..., 0]
plt.imshow(np.rot90(red[region, ...]))

plt.subplot(1, 2, 2)
plt.title("CC mask used for SNR computation")
plt.axis('off')
plt.imshow(np.rot90(mask_cc_part[region, ...]))
fig.savefig("CC_segmentation.png", bbox_inches='tight')


mean_signal = np.mean(data[mask_cc_part], axis=0)

from scipy.ndimage import binary_dilation
mask_noise = binary_dilation(mask, iterations=10)
mask_noise[..., :mask_noise.shape[-1]//2] = 1
mask_noise = ~mask_noise

save_nifti('mask_noise.nii.gz', mask_noise.astype(np.uint8), affine)

noise_std = np.std(data[mask_noise, :])
print('Noise standard deviation sigma= ', noise_std)

idx = np.sum(gtab.bvecs, axis=-1) == 0
gtab.bvecs[idx] = np.inf
axis_X = np.argmin(np.sum((gtab.bvecs-np.array([1, 0, 0]))**2, axis=-1))
axis_Y = np.argmin(np.sum((gtab.bvecs-np.array([0, 1, 0]))**2, axis=-1))
axis_Z = np.argmin(np.sum((gtab.bvecs-np.array([0, 0, 1]))**2, axis=-1))

for direction in [0, axis_X, axis_Y, axis_Z]:
    SNR = mean_signal[direction]/noise_std
    if direction == 0:
        print("SNR for the b=0 image is :", SNR)
    else:
        print("SNR for direction", direction, " ",
              gtab.bvecs[direction], "is :", SNR)
