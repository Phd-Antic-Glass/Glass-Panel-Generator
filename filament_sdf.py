# -----------------------------------------------------
# Quentin HUAN 07/2024
# 
# Script to generate the signed distance field of a glass filament along a trajectory.
# Uses drjit v0.4.6 for parallel evaluation on the GPU 
#  https://drjit.readthedocs.io/en/v0.4.6/
#
# The filament is made up several segments ( sdRoundCone() ) that are assembled to form the full sdf ( filament_sdf_from_trajectory() )
# We can build from the sdf a Refractive Index Field ( RIF_from_sdf() ) out of the occupency of the filament inside a voxel grid.
# The occupency is defined as the ratio between the volume of the filament and the volume of the current voxel.
# The volume is estimated by a simple MonteCarlo method ( compute_occupency() ).
# -----------------------------------------------------

import numpy as np
import drjit as dr
from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop
from matplotlib import pyplot as plt

# Filament segment sdf:
# p is a list points [ [x,y,z], ... , [x,y,z] ]
def sdRoundCone(p: Array3f, a, b, r1, r2):
    # sampling independent computations (only depend on shape)
    ba = b - a
    l2 = dr.dot(ba,ba)
    rr = r1 - r2
    a2 = l2 - rr*rr
    il2 = 1.0/l2

    # sampling dependant computations
    pa = p - a
    y = dr.dot(pa,ba)
    z = y - l2
    x2 = dr.norm( pa*l2 - ba*y )**2
    y2 = y*y*l2
    z2 = z*z*l2

    # single square root!
    k = dr.sign(rr)*rr*rr*x2
    return dr.select(dr.sign(z)*a2*z2>k, dr.sqrt(x2 + z2)*il2 - r2, dr.select(dr.sign(y)*a2*y2<k, dr.sqrt(x2 + y2)*il2 - r1, (dr.sqrt(x2*a2*il2)+y*rr)*il2 - r1) )

# sdf of a filament
# Ax,Ay,Az ,Bx,By,Bz the {start, end} point of each filament section
# Ra, Rb the {start, end} radii of each section
# N the number of sections
def sdFilament(p: Array3f, Ax, Ay, Az, Bx, By, Bz, Ra ,Rb, N):
    d = sdRoundCone(p, Array3f([0,0,0]), Array3f([0,0,0]), 0, 0)
    i = UInt32(0)

    loop = Loop("sdFilament", lambda: (d,i))
    while loop(i < N-1):
        a = Array3f(dr.gather(Float,Float(Ax),i) ,dr.gather(Float,Float(Ay),i),dr.gather(Float,Float(Az),i))
        b = Array3f(dr.gather(Float,Float(Bx),i) ,dr.gather(Float,Float(By),i),dr.gather(Float,Float(Bz),i))
        ra = dr.gather(Float,Float(Ra),i)
        rb = dr.gather(Float,Float(Rb),i)

        d = dr.minimum(d, sdRoundCone(p, a, b, ra, rb))
        i+=1
    return d

## compute filament SDF from provided trajectory in {r,theta,z} cartesian space
# Split the trajectory on discontinuities then contruct the filament SDF fron round cone primitives
# The SDF will be evaluated in a cubic domain of unit length to ensure proper antialising
def filament_sdf_from_trajectory(grid_shape, e, R,P,Z, Radius, MonteCarlo=False, next_x=0, next_y=0, next_z=0, radius_i=1):
    # domain size
    c = 1

    ## group trajectory in consecutive segments
    # cubic domain: scale R axis accordingly
    R1 = R[0:-2]
    R2 = R[1:-1]
    P1 = P[0:-2] 
    P2 = P[1:-1] 
    Z1 = Z[0:-2]
    Z2 = Z[1:-1]

    Radius1 = Radius[0:-2] 
    Radius2 = Radius[1:-1] 

    ## Split trajectory on two point distance discontinuities
    Distance = np.sqrt((R1-R2)**2 + (P1-P2)**2 + (Z1-Z2)**2)

    # split on discontinuities:
    # limit jump to 10 times the length of a voxel
    stepsize = (1.0*c)/np.max(grid_shape) 

    spliton =  np.where(np.diff(Distance) >= stepsize)[0]+2
    R1=np.split(R1, spliton)
    R2=np.split(R2, spliton)
    P1=np.split(P1, spliton)
    P2=np.split(P2, spliton)
    Z1=np.split(Z1, spliton)
    Z2=np.split(Z2, spliton)
    Radius1=np.split(Radius1, spliton)
    Radius2=np.split(Radius2, spliton)
    N = np.asarray([np.shape(Radius1[i])[0] for i in range(len(Radius1))])


    ## we compute the SDF with cubic voxels. 
    X = dr.linspace(Float,0,c,grid_shape[0], endpoint=False)
    Y = dr.linspace(Float,0,c,grid_shape[1], endpoint=False)
    Z = dr.linspace(Float,0,c,grid_shape[2], endpoint=False)



    # direct transform
    #  Z = Z*e + radius_i
    X,Y,Z = dr.meshgrid(X,Y,Z)
    if(MonteCarlo):
        X =X + next_x*(c/grid_shape[0])
        Y =Y + next_y*(c/grid_shape[1])
        Z =Z + next_z*(c/grid_shape[2])
        Z = Z*e + radius_i
    else:
        X = dr.linspace(Float,0,c,grid_shape[0], endpoint=False)
        Y = dr.linspace(Float,0,c,grid_shape[1], endpoint=False)
        Z = dr.linspace(Float,0, c,grid_shape[2], endpoint=False) - c/2
        #  Z = (Z-c*0.5)*e + radius_i + e/2
        #  Z = (Z-c*0.0)*0.5*e + radius_i
        #  Z = (Z-c*0.0) + radius_i
        X,Y,Z = dr.meshgrid(X,Y,Z)


    ## eval filament SDF on voxel grid
    # ignore first and last segments that tend to have rsp. much slower velocity / unfinished trajectories
    SDF = sdFilament([X,Y,Z], R1[1],P1[1],Z1[1],R2[1],P2[1],Z2[1],Radius1[1],Radius2[1],np.shape(Radius1[1])[0])
    # for each trajectory segments:
    for i in range(2,len(R1)-1):
        SDF = dr.minimum(SDF, sdFilament([X,Y,Z], R1[i],P1[i],Z1[i],R2[i],P2[i],Z2[i],Radius1[i],Radius2[i],np.shape(Radius1[i])[0]))
        i+=1

    return SDF

## compute the RIF distribution from provided SDF
def RIF_from_sdf(grid_shape, e, SDF, eta_base, eta_filament, antialiasing=True, mean_z=1):
    pixel_footprint=1.0/grid_shape[0]

    #  the SDF in computed with cubic voxels. The result will be scaled aferwards.
    occupency = compute_occupency(SDF, pixel_footprint, antialiasing)

    Eta = eta_base + eta_filament*occupency
    Eta = TensorXf(dr.ravel(Eta), shape=(grid_shape[0], grid_shape[1], grid_shape[2]*mean_z))

    return Eta

# compute voxel occupency from provided SDF 
def compute_occupency(SDF, pixel_footprint, antialiasing=True):
    occupency = 0*SDF
    if(antialiasing):
        SDF_pixel = SDF/pixel_footprint
        occupency = dr.clip(0.5-SDF_pixel, 0.0, 1.0)
    else:
        # no antialising
        occupency = dr.select(SDF < 0, 1.0, 0.0)
    return occupency
