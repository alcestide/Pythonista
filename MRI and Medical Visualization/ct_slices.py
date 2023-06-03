#!/usr/bin/env python3

import nibabel # library for working with NIfTI-1 Data Format
import numpy as np # numpy for image manipulation
from matplotlib import cm # color schemes for visualization
from matplotlib import pyplot as plt # library for visualization
filepath = 'volume_pt5/volume-44.nii'
imagedata=nibabel.load(filepath)
array = imagedata.get_fdata()
array = np.rot90(np.array(array))
print(array.shape)
f = plt.figure(figsize=(12,12))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.imshow(array[...,50].astype(np.float32), cmap=plt.cm.bone)
ax2.imshow(array[...,118].astype(np.float32), cmap=plt.cm.bone)
