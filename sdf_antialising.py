import numpy as np
from matplotlib import pyplot as plt

def sdf_circle(px, py, r):
    return np.sqrt(px**2 + py**2)-r


Res = 256
C = 1
pixel_footprint = C/Res
X = np.linspace(-C/2,C/2,Res)
Y = np.linspace(-C/2,C/2,Res)
X,Y = np.meshgrid(X,Y)

Z = sdf_circle(X,Y,0.3*C)

# simple circle drawing
Z_bin = Z > 0

# antialising
Z_pixel = Z/pixel_footprint
Z_bin_AA = 1-np.clip(0.5-Z_pixel, 0, 1)

plt.imshow(Z_bin, cmap="Greys")
plt.show()
plt.imshow(Z_bin_AA, cmap="Greys")
plt.show()








