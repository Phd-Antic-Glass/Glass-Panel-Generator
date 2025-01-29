import drjit as dr
from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

import numpy as np
from scipy.ndimage import geometric_transform

import cv2
from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

from matplotlib import pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


def dilate(img_path):
    Img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(Img.shape)
    N = Img.shape[0]

    tensor_shape = (N,N,1,1)
    tex_heightmap = Texture3f(TensorXf(Float(np.ravel(Img)), shape=tensor_shape), wrap_mode=dr.WrapMode.Mirror)

    xx = dr.linspace(Float, 0, 1, N)
    yy = dr.linspace(Float, 0, 1, N)
    X, Y = dr.meshgrid(xx,yy)

    uv = Array3f(0, X, Y) + Array3f(0, 0.5*(X-0.5) , 0)

    Img_warp = tex_heightmap.eval(uv)[0]



    Img_warp = TensorXf(Img_warp, Img.shape).numpy()
    plt.imshow(Img_warp)
    plt.show()


dilate("./heightmaps/cloud_blur.exr")












