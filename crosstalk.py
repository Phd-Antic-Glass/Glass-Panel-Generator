import numpy as np
import drjit as dr
from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop
import bubble_generator as bgen


def sdDisplacement(p: Array3f, H1, H2, tex_heightmap_front, tex_heightmap_back):
    uv = Array3f(0,p.x,p.y)

    return dr.select(p.z>0, tex_heightmap_front.eval(uv)[0]*H1, tex_heightmap_back.eval(uv)[0]*H2)

def sdBox(p: Array3f, b: Array3f):
    q = dr.abs(p) - b
    return dr.norm(dr.maximum(q, Array3f(0))) + dr.minimum(dr.maximum(q.x, dr.maximum(q.y, q.z)), 0.0)
    #  return -p.z + b.z

def smooth_union(a, b, k):
    k *= 4.0
    h = dr.maximum( k-dr.abs(a-b), 0.0 )/k
    return dr.minimum(a,b) - h*h*k*(1.0/4.0)

def sdPanel(p: Array3f, panel_dimension, H1, H2, tex_heightmap_front, tex_heightmap_back, tex_SDF_bubble, tex_SDF_RIF, no_surface, no_bubbles, no_chords):
    # panel is centered on z axis
    p_center = p - Array3f(panel_dimension[0], panel_dimension[1], 0) * 0.5

    displacement = sdDisplacement(p, H1, H2, tex_heightmap_front, tex_heightmap_back)

    box_dim = Array3f(panel_dimension) * 0.5 + dr.select(no_surface, 0.0, displacement)

    box_d = sdBox(p_center, box_dim)

    sdf_bubble = tex_SDF_bubble.eval([p.z,p.x,p.y])[0]
    sdf_chords = tex_SDF_RIF.eval([p.z,p.x,p.y])[0]

    crosstalk_bubble = dr.select(no_bubbles, box_d,smooth_union(box_d, sdf_bubble, 0.001))
    crosstalk_bubble_chords = dr.select(no_chords, crosstalk_bubble, smooth_union(crosstalk_bubble, sdf_chords, 0.05))

    return crosstalk_bubble_chords

def trace_heightmap(o: Array3f, d: Array3f, panel_dimension, H1, H2, tex_heightmap_front, tex_heightmap_back, tex_SDF_bubble, tex_SDF_RIF, no_surface, no_bubbles, no_chords):
    t = Float(0.0)
    D = Float(0.0)
    N = UInt32(500)
    i = UInt32(0)
    loop = Loop("sphere_trace", lambda: (t,D,i,N))
    while loop(i<N):
        D = sdPanel(o + t*d, panel_dimension, H1, H2,tex_heightmap_front, tex_heightmap_back, tex_SDF_bubble, tex_SDF_RIF, no_surface, no_bubbles, no_chords)
        t += D
        i+=1
    return t

## interaction bubbles -> heightmap
def cross_talk(heightmap_front, heightmap_back, H1, H2, e, bubble_distrib, bubble_shape, radius_range, panel_dimension, sdf_bake_dimension, SDF_RIF, no_surface, no_bubbles, no_chords):
    SDF_bubble = bgen.bake_sdf_bubbles(bubble_distrib, panel_dimension, sdf_bake_dimension, MonteCarlo=False)

    tensor_shape = (sdf_bake_dimension[0],sdf_bake_dimension[1],sdf_bake_dimension[2],1)
    tex_SDF_bubble = Texture3f(TensorXf(SDF_bubble, shape=tensor_shape))
    tex_SDF_RIF = Texture3f(TensorXf(SDF_RIF, shape=tensor_shape))

    tensor_shape = (heightmap_front.shape[0],heightmap_front.shape[1],1,1)
    tex_heightmap_front = Texture3f(TensorXf(Float(np.ravel(heightmap_front)), shape=tensor_shape))

    tensor_shape = (heightmap_front.shape[0],heightmap_front.shape[1],1,1)
    tex_heightmap_back = Texture3f(TensorXf(Float(np.ravel(heightmap_back)), shape=tensor_shape), wrap_mode=dr.WrapMode.Repeat)


    # front
    X = dr.linspace(Float,0,panel_dimension[0],heightmap_front.shape[0], False)
    Y = dr.linspace(Float,0,panel_dimension[1],heightmap_front.shape[1], False)
    X,Y = dr.meshgrid(X,Y)
    o = [X,Y,panel_dimension[2]]

    dX = dr.zeros(Float,heightmap_front.shape[0])
    dY = dr.zeros(Float,heightmap_front.shape[1])
    dX,dY = dr.meshgrid(dX,dY)
    d = [dX, dY, -1]

    heightmap_front_crosstalk = trace_heightmap(Array3f(o), Array3f(d), panel_dimension, H1, H2, tex_heightmap_front, tex_heightmap_back, tex_SDF_bubble, tex_SDF_RIF, no_surface, no_bubbles, no_chords)
    heightmap_front_crosstalk = dr.abs(o[2] + heightmap_front_crosstalk*d[2]) - panel_dimension[2]*0.5

    # back
    o = [X,Y,-panel_dimension[2]]
    d = [dX, dY, 1]

    heightmap_back_crosstalk = trace_heightmap(Array3f(o), Array3f(d), panel_dimension, H1, H2,  tex_heightmap_front, tex_heightmap_back,tex_SDF_bubble, tex_SDF_RIF, no_surface, no_bubbles, no_chords)

    heightmap_back_crosstalk = dr.abs(o[2] + heightmap_back_crosstalk*d[2]) - panel_dimension[2]*0.5
    heightmap_front_crosstalk = TensorXf(heightmap_front_crosstalk, heightmap_front.shape).numpy()
    heightmap_back_crosstalk = TensorXf(heightmap_back_crosstalk, heightmap_back.shape).numpy()
    return heightmap_front_crosstalk, heightmap_back_crosstalk 
