# Sample MR Cardiovascular Aorta FLASH 3D cor
# DCM image import and visualization using Plotly

import imageio.v2 as iio
import scipy.ndimage as ndi
import numpy as np
import plotly.express as px
from skimage import io
import plotly

# Change this to your default browser
plotly.io.renderers.default = 'chromium'

# Load .dcm file
aorta = iio.imread('IMA05.IMA')

fig = px.imshow(aorta,
                title="<b>MAGNETOM Symphony - Cardiovascular</b> \
                Aorta FLASH 3D cor Institute for Micro Therapy, Bochum, Germany",
                color_continuous_scale='aggrnyl',
                zmin=-10,
                zmax=250)

fig.show()
