import numpy as np
from matplotlib import pyplot as plt

#######################################################################
#                               CONFIG                                #
#######################################################################
N = 10*5800 # nb of steps
dt = 1e-3 # time step
drag = 0.4 # drag scaling
M = 0.025 # mass of the particle
turbulence_intensity = 1.4
seed = 3

eta_base = 1.54
eta_delta = 0.05

e = 0.3 # tube shell width
radius_i = 1 # inner radius if the tube domain
radius_o = radius_i + e # outer radius if the tube domain
Zmax = 1

R = np.zeros(N) # polar coords radius
P = np.zeros(N) # polar coords angle
Z = np.zeros(N) # polar coords height
T = np.zeros(N) # Time 

# initial conditions
R0 = radius_i - e*0.5 
P0 = 0
Z0 = 0

R_dot_0 = 0.1
P_dot_0 = 1.0
Z_dot_0 = 0.1

R1 = R_dot_0*dt +R0
P1  = P_dot_0*(1/R1)*dt + P0
Z1 = Z_dot_0*dt +Z0

R[0] = R0
R[1] = R1
P[0] = P0
P[1] = P1
Z[0] = Z0
Z[1] = Z1
T[0] = 0
T[1] = dt

#######################################################################
#                            PERLIN NOISE                             #
#######################################################################

## PERLIN NOISE Generator
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
from scipy.interpolate import RegularGridInterpolator as rgi

np.random.seed(seed)
noise_res = 128
noise_freq = 4
noise = generate_fractal_noise_3d(
    (noise_res, noise_res, noise_res), (noise_freq,noise_freq,noise_freq ), 4, tileable=(True, True, True)
)

## Interpolator 
X_noise = np.linspace(-2*radius_o,2*radius_o, noise_res)
Y_noise = X_noise
Z_noise = X_noise
interp = rgi((X_noise,Y_noise,Z_noise), noise)

print("Noise generation done.")
#######################
#  Force definitions  #
#######################

## radial force
def Force_R(r, t):
    F = 0.1*(np.sin(0.9*t) + 0.1)
    A = 0.1
    if(r < radius_i):
        F = F + A
    if(r > radius_o - e*0.5):
        F = F - A
    return F

## angular force
def Force_P(p, t):
    return 1.0*(np.sin(0.1*(2**0.5)*t)**2 + 0.1 + np.cos(0.7*t)**2)

## quick and dirty explicit newton's method
for i in range(2, N):
    R_kp2 = R[i+0]
    R_kp1 = R[i-1]
    R_k   = R[i-2]

    P_kp2 = P[i+0]
    P_kp1 = P[i-1]
    P_k   = P[i-2]

    Z_kp2 = Z[i+0]
    Z_kp1 = Z[i-1]
    Z_k   = Z[i-2]

    t = i*dt # time

    ## cartesian coords
    x = R_k*np.cos(P_k)
    y = R_k*np.sin(P_k)
    z = Z_k

    # noise at point (x,y,z)
    noise = interp(np.clip(np.array([x,y,np.mod(z,1)]).T, -2*radius_o, 2*radius_o) ) * turbulence_intensity

    V = np.sqrt(((R_kp1-R_k)/dt)**2 + (R_kp1*(P_kp1-P_k)/dt)**2 + ((Z_kp1-Z_k)/dt)**2) # speed magnitude
    V = np.clip(V, -1, 1)

    ## update point kp2
    R_kp2 = -( - 2*R_kp1 + R_k - R_k*(P_kp1**2 - 2*P_kp1*P_k + P_k**2) ) + (Force_R(R_k, t) - drag*V*(((R_kp1-R_k)/dt)))*(dt*dt)/M
    #  2*( R_kp1-R_k )*(P_kp1-P_k)/(dt*dt) + R_k*(P_kp2 - 2*P_kp1 + P_k)/(dt*dt) = F
    P_kp2   = (2*P_kp1 - P_k ) + ( (Force_P(P_k, t) + noise*2 - drag*V*(R_kp1*(P_kp1-P_k)/dt) )  - 2*( R_kp1-R_k )*(P_kp1-P_k))*(dt*dt) /(R_k*M)
    #  R_k*(P_kp2 - 2*P_kp1 + P_k)/(dt*dt) = ( F - 2*( R_kp1-R_k )*(P_kp1-P_k)/(dt*dt)  )/(R_k)*dt*dt
    #  (Z_kp2 - 2*Z_kp1 + Z_k)/(dt*dt) = 0
    Z_kp2 = (0.1 -drag*V*((Z_kp1-Z_k)/dt))*(dt*dt)/M + 2*Z_kp1 - Z_k

    # output updated point
    R[i] = R_kp2
    P[i] = P_kp2
    Z[i] = Z_kp2
    T[i] = t

## wrap P and Z coords along (resp.) [0,2pi] and [0,Zmax]
Z = np.mod(Z, 1)
P = np.mod(P, 2*np.pi)
print("Simulation done.")
#######################################################################
#                                GRAPH                                #
#######################################################################

# cartesian coordinates
X = R*np.cos(P)
Y = R*np.sin(P)

fig, axs = plt.subplots(2,2)

alpha = 0.1

## Top view
ax = axs[0,0]
ax.set_xlim([-radius_o * 1.5, radius_o*1.5])
ax.set_ylim([-radius_o * 1.5, radius_o*1.5])
# draw trajectory
ax.plot(X,Y, color=(0,0,1,alpha*4))
# draw limits cylinder
circle_inner = plt.Circle((0, 0), radius_i, color='r', fill=False, zorder=2)
circle_outer = plt.Circle((0, 0), radius_o, color='r', fill=False, zorder=2)
ax.add_patch(circle_inner)
ax.add_patch(circle_outer)

## Panel outpul
ax = axs[0,1]
ax.set_xlim([0, 2*np.pi])
ax.set_ylim([0,1])
ax.plot(P, Z, ".", color=(0,0,1,alpha))


#######################################################################
#                             Convolution                             #
#######################################################################
eta_shape = (1024, 1024, 2)
radius_chords = 0.002
etamap = np.ones(eta_shape)*eta_base

# normalize simulation data by panel dimensions
def normalize_coords(R, P, Z, R_min, R_max, Z_min, Z_max):
    R_n = ((R-R_min)/(R_max-R_min))
    P_n = ((P-0)/(2*np.pi))
    Z_n = ((Z-Z_min)/(Z_max-Z_min))
    return [R_n, P_n, Z_n]

# convert simulation data to etamap cell id (int)
# panel dimension is {1, 1, e}
def to_cell_id(R, P, Z, grid_shape, R_min, R_max, Z_min, Z_max):
    R_n, P_n, Z_n = normalize_coords(R,P,Z, R_min, R_max, Z_min, Z_max)
    R_ijk = np.asarray( np.clip( np.floor(R_n*grid_shape[2]) ,0,grid_shape[2]-1), dtype=np.uint32 ) # R corresponds to z
    P_ijk = np.asarray( np.clip( np.floor(P_n*grid_shape[0]) ,0,grid_shape[0]-1), dtype=np.uint32 ) # P corresponds to x
    Z_ijk = np.asarray( np.clip( np.floor(Z_n*grid_shape[1]) ,0,grid_shape[1]-1), dtype=np.uint32 ) # Z corresponds to y

    return [R_ijk, P_ijk, Z_ijk]


R_ijk, P_ijk, Z_ijk = to_cell_id(R,P,Z, eta_shape, radius_i, radius_o, 0, Zmax)

## draw etamap along trajectory
etamap[P_ijk, Z_ijk, R_ijk] = eta_base + eta_delta


from scipy.ndimage import gaussian_filter
sigma = (radius_chords*eta_shape[0], radius_chords*eta_shape[1], radius_chords*(eta_shape[2]/e))
etamap = gaussian_filter(etamap, sigma)

print("convolution done")   
np.save("RIF.npy", etamap)

img = np.reshape(etamap[:,:,0], (eta_shape[0], eta_shape[1]))
axs[1,0].imshow(img, cmap='plasma', vmin=eta_base, vmax=2.0, origin="lower")

img = np.reshape(etamap[:,:,1], (eta_shape[0], eta_shape[1]))
axs[1,1].imshow(img, cmap='plasma', vmin=eta_base, vmax=2.0, origin="lower")







plt.show()
