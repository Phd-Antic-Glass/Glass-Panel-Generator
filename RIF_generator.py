# -----------------------------------------------------
# Quentin HUAN 06/2024
# 
# Generate a {grid_shape[0], grid_shape[1], grid_shape[2]}x1 array, each voxel stores the Refractive index field at that point. Suitable for simulating chords on blown cylinder glass.
# Uses drjit v0.4.6 for parallel evaluation on the GPU 
#  https://drjit.readthedocs.io/en/v0.4.6/
#
# Molten glass filaments get coiled around the cylinder as it is blowned.
# We emulate this phenomenon by simulating the trajectory  in cylindrical space of a point of mass M subject to an angular force P(p, t) and a radial force R(p, t).
# The particle is also subject to a drag force of type -drag*V and an angular perturbation term noise*turpulence where noise is a 3 dimensional scalar field defined by a perlin noise.
# The simulation of the trajectory follows a simple explicit euler scheme.
# The angular component is wraped every 2*pi units, and the height Z axis is wrapped every 1 units.
#
# The particle evolves around two cylinders of radius radius_i and radius_i+e which correspond respectively to the front and back face of the blown panel. 
#
# The RIF is then generated constructing the SDF of the the glass filaments and estimating its volume occupency inside each voxels.
# Voxel grid axis X is the angular component of the trajectory, Y its vertical component and Z its radial component. 
# These coordinates are then interpreted as the {X,Y,Z} component of cartesian space.
# -----------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import RegularGridInterpolator as rgi

from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

from filament_sdf import *


#######################
#  Force definitions  #
#######################

## radial force
def Force_R(r, t, radius_i, radius_o):
    # amplitude
    A = 5.5
    e = radius_o-radius_i
    F = -A*(r-(radius_i))
    F = F -A*0.25*(r-(radius_o-0.25*e))
    return F

## angular force
def Force_P(p, t):
    #  return 1.0*(np.sin(0.1*(2**0.5)*t)**2 + 0.1 + np.cos(0.7*t)**2)
    return np.cos(0.7*t)**2

## angular force
def Force_Z(p, t):
    #  return 1.0*(np.sin(0.1*(2**0.5)*t)**2 + 0.1 + np.cos(0.7*t)**2)
    return 0.1*np.cos(0.05*t)


# Compute the trajectory of the filament.
# Inputs: 
# - radius_i corresponds to interrior radius of the blown cylinder 
# - e corresponds to the bubble thickness
#
# Outputs:
# - trajectory coordinates {p_i,z_i,r_i / i \in [0,N]}
# - Velocity module {norm(v_i) / i \in [0,N]}
# - Time {\tau * i / i \in [0,N]} 
def simulate_trajectory(seed, radius_i, e):
    ## euler method parameters
    N = 10*580*1 # nb of integration steps
    dt = 1e-2 # time step
    drag = 0.65 # drag coef
    M = 0.025 # mass of the particle
    turbulence_intensity = 0.1

    ## inner and outer gathering limits
    # inner radius if the tube domain
    radius_i = 0.25
    # outer radius if the tube domain
    radius_o = radius_i + e
    # wrap Z axis over itself after Zmax units
    Zmax = 1 

    ## trajectory arrays
    R = np.zeros(N) # polar coords radius
    P = np.zeros(N) # polar coords angle
    Z = np.zeros(N) # polar coords height
    T = np.zeros(N) # Time 
    V = np.zeros(N) # Speed 

    ## PERLIN NOISE Generator
    np.random.seed(int(seed))
    noise_res = 128
    noise_freq = 4
    noise = generate_fractal_noise_3d(
        (noise_res, noise_res, noise_res), (noise_freq,noise_freq,noise_freq ), 4, tileable=(True, True, True)
    )

    # Interpolator 
    X_noise = np.linspace(-2*radius_o,2*radius_o, noise_res)
    Y_noise = X_noise
    Z_noise = X_noise
    interp = rgi((X_noise,Y_noise,Z_noise), noise)

    ## initial conditions
    # positions
    R[0] = radius_i 
    P[0] = 0
    Z[0] = 0
    V[0] = 0

    # speeds
    R_dot_0 = 0.1
    P_dot_0 = 1.0
    Z_dot_0 = 0.1

    R[1] = R_dot_0*dt +R[0]
    P[1]  = P_dot_0*(1/R[1])*dt + P[0]
    Z[1] = Z_dot_0*dt +Z[0]
    V[1] = np.sqrt(((R[1]-R[0])/dt)**2 + (R[1]*(P[1]-P[0])/dt)**2 + ((Z[1]-Z[0])/dt)**2) # speed magnitude


    # fill Time array
    T[0] = 0
    T[1] = dt

    ##  Explicit euler method:
    # quick and dirty ... 
    for i in range(2, N):
        print(" RIF generator (filament trajectory): ", ((i+1)/N)*100, "%", end="\r")
        ## get current and the two previous points
        # radial
        R_kp2 = R[i+0]
        R_kp1 = R[i-1]
        R_k   = R[i-2]
        # angular
        P_kp2 = P[i+0]
        P_kp1 = P[i-1]
        P_k   = P[i-2]
        # vertical
        Z_kp2 = Z[i+0]
        Z_kp1 = Z[i-1]
        Z_k   = Z[i-2]

        ## get current time
        t = i*dt

        ## cartesian coords
        x = R_k*np.cos(P_k)
        y = R_k*np.sin(P_k)
        z = Z_k

        ## read noise at point (x,y,z)
        noise = interp(np.clip(np.array([x,y,np.mod(z,1)]).T, -2*radius_o, 2*radius_o) ) * turbulence_intensity

        ## speed vector magnitude
        v = np.sqrt(((R_kp1-R_k)/dt)**2 + (R_kp1*(P_kp1-P_k)/dt)**2 + ((Z_kp1-Z_k)/dt)**2) # speed magnitude

        # bounds speed to avoid that the simulation blows up 
        #(a bit hacky but can make the parameters easier to tune)
        #  v = np.clip(v, -1, 1)

        ## update point kp2
        # radial
        R_kp2 = -( - 2*R_kp1 + R_k - R_k*(P_kp1**2 - 2*P_kp1*P_k + P_k**2) ) + (Force_R(R_k, t, radius_i, radius_o) - drag*v*(((R_kp1-R_k)/dt)))*(dt*dt)/M
        # angular
        P_kp2   = (2*P_kp1 - P_k ) + ( (Force_P(P_k, t) + noise - drag*v*(R_kp1*(P_kp1-P_k)/dt) )  - 2*( R_kp1-R_k )*(P_kp1-P_k))*(dt*dt) /(R_k*M)
        # vertical
        Z_kp2 = (Force_Z(P_k, t) + noise -drag*v*((Z_kp1-Z_k)/dt))*(dt*dt)/M + 2*Z_kp1 - Z_k

        ## write updated point in trajectory arrays
        R[i] = R_kp2
        P[i] = P_kp2
        Z[i] = Z_kp2
        T[i] = t
        V[i] = v

    ## output trajectory arrays
    return P,R,Z,T,V


# - grid_shape is the RIF [x,y,z] resolution
# - eta_base is the refractive index of the panel if it was perfectly homogeneous (typical value=1.54)
# - delta_eta defines the maximum amplitude of the refractive index heterogeneities (typical value-0.05)
# - debug=True activates the visualization of the trajectory and the corresponding RIF using matplotlib
# - e corresponds to the panel thickness
# - chord radius is the blurring factor to apply to the trajectory generating the chords
def generate_RIF(grid_shape,sdf_bake_dimension, eta_base, delta_eta, seed, e, chord_throughtput, save_RIF_to_file=True, debug=True):
    chord_throughtput = 0.00001
    radius_i = 0.25
    radius_o = radius_i + e

    ## compute trajectory of the contact point btwn filament and the cylindrical bubble
    P,R,Z,T,V = simulate_trajectory(seed, radius_i, e)

    # wrap P and Z coords along (resp.) [0,2pi] and [0,Zmax]
    P = np.mod(P, 2*np.pi)
    Z = np.mod(Z, 1)

    # compute radius of the filament from material chord_throughtput and contact point speed
    radius = np.clip(np.sqrt(chord_throughtput/(np.pi*V+0.00001)), 0, 0.1)

    ## compute RIF:
    # parallel random generator
    rngXYZ = PCG32(dr.prod(grid_shape))

    # cull some points for faster SDF evaluation
    downsample_trajectory = 4
    P_=P[0::downsample_trajectory]
    R_=R[0::downsample_trajectory] 
    R = R - radius_i -e/2
    Z_=Z[0::downsample_trajectory]
    radius=radius[0::downsample_trajectory]

    ## MonteCarlo RIF estimation
    # init
    SDF = filament_sdf_from_trajectory(grid_shape, e, P_/(2*np.pi),Z_,R_, radius, MonteCarlo=True, next_x=rngXYZ.next_float32(),next_y=rngXYZ.next_float32(), next_z=rngXYZ.next_float32(), radius_i=radius_i)
    RIF = RIF_from_sdf(grid_shape,e,SDF,eta_base, delta_eta, antialiasing=True)

    # mc estimation
    i = UInt32(0)
    # you may need more monte carlo samples to remove noise (typical value for noise free: 10000)
    spvoxel = UInt32(100)
    print("RIF generator (filament density estimation): ", spvoxel, "samples per voxels requiered")
    loop = Loop("MC occupancy", lambda: (i, SDF, RIF,rngXYZ))
    while loop(i < spvoxel):
        SDF_ = filament_sdf_from_trajectory(grid_shape, e, P_/(2*np.pi),Z_,R_, radius, MonteCarlo=True, next_x=rngXYZ.next_float32(),next_y=rngXYZ.next_float32(), next_z=rngXYZ.next_float32(), radius_i=radius_i)
        RIF += RIF_from_sdf(grid_shape,e,SDF_,eta_base, delta_eta, antialiasing=True)
        i += 1
    # mean
    RIF = RIF/(spvoxel+1)


    ## SDF baking for chord crosstalk
    SDF = filament_sdf_from_trajectory(sdf_bake_dimension, e, P_/(2*np.pi),Z_,R_, radius, MonteCarlo=False, next_x=rngXYZ.next_float32(),next_y=rngXYZ.next_float32(), next_z=rngXYZ.next_float32(), radius_i=radius_i)

    ## migrate data back to CPU memory
    RIF = np.asarray(RIF)
    print(np.max(RIF))
    SDF = np.asarray(SDF)

    ## save to file if needed
    if(save_RIF_to_file):
        np.save("RIF.npy", RIF)

    ## optional visualization of the RIF and the trajectory
    if(debug):
        visualize_simulation(grid_shape,radius_i,radius_o,RIF,R,P,Z,eta_base,delta_eta)

    return RIF, SDF

### draw the simulated trajectory in {P,R} and {P,Z} planes and the first and last voxel layers of the RIF
#def visualize_simulation(grid_shape, radius_i, radius_o, RIF, R, P, Z, eta_base, delta_eta):
#    # cartesian coordinates
#    X = R*np.cos(P)
#    Y = R*np.sin(P)
#
#    fig, axs = plt.subplots(2,1)
#
#    # trajectory transparency
#    alpha = 0.1
#
#    ## Top view {X, Y}
#    ax = axs[0,0]
#    ax.set_xlim([-radius_o * 1.5, radius_o*1.5])
#    ax.set_ylim([-radius_o * 1.5, radius_o*1.5])
#
#    # draw trajectory
#    ax.plot(X,Y, color=(0,0,1,alpha*4))
#
#    # draw limits cylinder
#    circle_inner = plt.Circle((0, 0), radius_i, color='r', fill=False, zorder=2)
#    circle_outer = plt.Circle((0, 0), radius_o, color='r', fill=False, zorder=2)
#    ax.add_patch(circle_inner)
#    ax.add_patch(circle_outer)
#
#    ## Panel space trajectory
#    ax = axs[0,1]
#    ax.set_xlim([0, 2*np.pi])
#    ax.set_ylim([0,1])
#
#    ax.plot(P, Z, ".", color=(0,0,1,alpha))
#
#    ## show generated RIF (first and last voxel layers)
#    RIF_mean = np.mean(RIF, axis=2)
#    #  img = np.reshape(RIF[:,:,0], (grid_shape[0], grid_shape[1]))
#    axs[1,0].imshow(RIF_mean, cmap='viridis',vmin=eta_base, vmax=eta_base+delta_eta,  origin="lower")
#
#    img = np.reshape(RIF[:,:,-1], (grid_shape[0], grid_shape[1]))
#    axs[1,1].imshow(img, cmap='plasma', vmin=eta_base, vmax=eta_base+delta_eta, origin="lower")
#
#    plt.show()

## draw the simulated trajectory in {P,R} and {P,Z} planes and the first and last voxel layers of the RIF
def visualize_simulation(grid_shape, radius_i, radius_o, RIF, R, P, Z, eta_base, delta_eta):
    e = np.abs(radius_i-radius_o)
    # cartesian coordinates
    X = ( R + radius_i + 0.5*e )*np.cos(P)
    Y = ( R + radius_i + 0.5*e )*np.sin(P)

    # trajectory transparency
    alpha = 0.1

    ## Top view {X, Y}
    plt.xlim([-radius_o * 1.5, radius_o*1.5])
    plt.ylim([-radius_o * 1.5, radius_o*1.5])

    # draw trajectory
    plt.plot(X,Y, color=(0,0,1,alpha*4))

    # draw limits cylinder
    circle_inner = plt.Circle((0, 0), radius_i, color='r', fill=False, zorder=2)
    circle_outer = plt.Circle((0, 0), radius_o, color='r', fill=False, zorder=2)
    plt.gca().add_patch(circle_inner)
    plt.gca().add_patch(circle_outer)

    plt.savefig("cylindrical_trajectory.pdf")
    plt.show()

    ## Panel space trajectory
    plt.xlim([0, 2*np.pi])
    plt.ylim([0,1])

    plt.plot(P, Z, ".", color=(0,0,1,alpha))
    plt.savefig("panel_trajectory.pdf")
    plt.show()

    ## Panel radial trajectorr
    plt.xlim([0, 2*np.pi])
    plt.ylim([-e,e])
    plt.plot(P, R, ".", color=(0,0,1,alpha))
    plt.plot([0, 2*np.pi], [-e/2, -e/2], "-r") 
    plt.plot([0, 2*np.pi], [e/2, e/2], "-r") 
    plt.savefig("panel_trajectory_PR.pdf")
    plt.show()
    ## show generated RIF (first and last voxel layers)
    RIF_mean = np.mean(RIF, axis=2)
    #  img = np.reshape(RIF[:,:,0], (grid_shape[0], grid_shape[1]))
    plt.imshow(RIF_mean, cmap='viridis',vmin=eta_base, vmax=eta_base+delta_eta,  origin="lower")
    plt.savefig("RIF_slice.pdf")
    plt.show()

