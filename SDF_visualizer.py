import drjit as dr
from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

N=1000
X = dr.linspace(0,1,N)
Y = dr.linspace(0,1,N)
Z = dr.ones(N)*0

X,Y = dr.meshgrid(X,Y)


