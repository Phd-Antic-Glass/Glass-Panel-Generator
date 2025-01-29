from zlib import Z_DEFAULT_COMPRESSION
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import ones

from scipy.interpolate import RegularGridInterpolator as rgi
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

from os import environ
from scipy import integrate
import time
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

def polar2Cart(r,theta):
    return (r*np.cos(theta),r*np.sin(theta))

def cart2Polar(x,y):
    return (np.sqrt(x**2+ y**2),np.arctan2(y,x))

def gaussian(x,sig,mu):
    r = np.exp((-0.5)*(((x-mu)/sig)**2))
    A = np.exp((-0.5)*(((mu-mu)/sig)**2))
    return  r/A
#-----------------------------------------------------------------------------
#                                   PARAMETERS
#-----------------------------------------------------------------------------
N = 1024
bit_depth=16
maxHeight = 0.4

x_offset = 0.5
y_offset = 0.5

# sine waves
SineAmplitude = 0.25*1
SineFrequency = 100
SineFalloffStrenght = 0.1
SineWarp = 0.15

# cone profile
ConeHeight = 0.01

# boudine
RayonBoudine = 0.2
TailleBoudine = (1-(SineAmplitude)-ConeHeight)

#perlin noise parameters
seed = 8

#-----------------------------------------------------------------------------
#                                   Setup
#-----------------------------------------------------------------------------
# Perlin Noise
np.random.seed(int(seed))

noise_res = 256
noise_freq = 2
noise = generate_fractal_noise_2d(
    (noise_res, noise_res), (noise_freq,noise_freq,noise_freq ), 4, tileable=(True, True)
)
print(noise.shape)

X_noise = np.linspace(-1,1, noise_res)
Y_noise = X_noise
Z_noise = X_noise
interp = rgi((X_noise,Y_noise), noise)
print(interp)


H = np.zeros((N,N),dtype=np.float32)

scaleVariation = 0
Xscale = np.random.uniform(1-scaleVariation,1+scaleVariation)
Yscale = np.random.uniform(1-scaleVariation,1+scaleVariation)
X = np.linspace(-1,1,N,dtype=np.float32) * Xscale
Y = np.linspace(-1,1,N,dtype=np.float32) * Yscale
X, Y = np.meshgrid(X, Y)

P = interp((X,Y))
print(P)


R = np.sqrt(X**2 + Y**2)

#-----------------------------------------------------------------------------
#                            HeightMap generation
#-----------------------------------------------------------------------------
def h_function(R):

    # F = np.sin(f*R + 2*Phase)
    sine = (np.cos(SineFrequency*(R+P*SineWarp)))**2
    falloff = (gaussian(R,0.5*SineFalloffStrenght,0)) - (gaussian(R,0.25*SineFalloffStrenght,0))
    boudine = (gaussian(R,0.5*RayonBoudine,0))*TailleBoudine

    # SUM
    return SineAmplitude*falloff*sine + boudine + (1-R)*ConeHeight

print("HeightMap generation...")
H = h_function(R)
    
print("DONE\n")


exportPath = './heightmaps/'

path = exportPath + "crown_H_"+str(N)+".exr"
H = np.asarray(H, dtype=np.float32)
cv2.imwrite(path, H.T)
print("saved HeightMap in " + path)

#-----------------------------------------------------------------------------
#                               HeightMap ploting
#-----------------------------------------------------------------------------
plt.axis('equal')
T = linspace(-1,1,N)

plt.plot(T,H[:,int(N/2)])
plt.savefig("crown_section.pdf")
plt.show()
