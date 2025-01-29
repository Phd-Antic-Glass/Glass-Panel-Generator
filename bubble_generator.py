import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import product
import drjit as dr
from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

from matplotlib import pyplot as plt


# Generate a {grid_shape[0], grid_shape[1], grid_shape[2]}x4 array, each voxel stores:
#   - bubble position {x,y,z}
#   - bubble radius {r}
#
# Inputs: 
# - radius_range {radius_min, radius_max}
# - density defines the number of bubbles: density := number_of_bubbles / number_of_voxels
# - panel_dimension [x,y,z] with z the width of the panel.
#
# Note: bubbles can be cut at voxel boundaries
def random_bubble_distrib_simplest(grid_shape, radius_range, density, panel_dimension=[1,1,0.01], seed=0):
    rng = np.random.default_rng(seed=42)
    number_of_bubbles = int(density*grid_shape[0]*grid_shape[1]*grid_shape[2])
    e = panel_dimension[2]

    ## Positions
    # positions between 0 and 1
    Positions = np.random.random((number_of_bubbles,3))
    # positions.z in [0, e]
    #  Positions[:,2] = radius_range[0] + (Positions[:,2])*((e-radius_range[1])-radius_range[0])
    Positions[:,2] = np.random.uniform(radius_range[1], e-radius_range[1], number_of_bubbles)

    ## compute bubble voxel identifiers
    Identifier = np.zeros((number_of_bubbles,3), dtype=np.uint32)
    Identifier[:, 0:2] = np.floor(Positions[:, 0:2]*grid_shape[0:2])
    Identifier[:, 2] = np.floor(((Positions[:, 2])/e)*grid_shape[2])

    ## radius between radius_min and radius_max
    Radius = radius_range[0] + (rng.random((number_of_bubbles,1))*(radius_range[1]-radius_range[0]))

    ## create and fill voxel grid
    Voxel_grid = np.zeros(grid_shape+(4,))

    Positions[:,2] = Positions[:,2] - 0.5*e
    bubbles_data = np.append(Positions, Radius, axis=1)
    bubble_id = np.asarray([i for i in range(number_of_bubbles)], dtype=np.uint32)

    # for each bubble, fill corresponding voxel
    Voxel_grid[Identifier[bubble_id,0], Identifier[bubble_id,1], Identifier[bubble_id,2], :] = bubbles_data[bubble_id]

    return Voxel_grid

# Generate a {grid_shape[0], grid_shape[1], grid_shape[2]}x4 array, each voxel stores:
#   - bubble position {x,y,z}
#   - bubble radius {r}
#
# Inputs: 
# - radius_range {radius_min, radius_max}
# - density defines the number of bubbles: density := number_of_bubbles / number_of_voxels
# - panel_dimension [x,y,z] with z the width of the panel.
#
# sort the bubbles and put them inside the voxel grid one at a time to avoid conflicts
def random_bubble_distrib(grid_shape, radius_range, density, panel_dimension=[1,1,0.01], seed=0):
    rng = np.random.default_rng(seed=42)
    number_of_bubbles = int(density*grid_shape[0]*grid_shape[1]*grid_shape[2])
    e = panel_dimension[2]

    ## Positions
    # positions between 0 and 1
    Positions = np.random.random((number_of_bubbles,3))
    # positions.z in [0, e]
    Positions[:,2] = radius_range[0] + (Positions[:,2])*((e-radius_range[1])-radius_range[0])
    ## radius between radius_min and radius_max
    Radius = radius_range[0] + (rng.random((number_of_bubbles,1))*(radius_range[1]-radius_range[0]))

    ## sort bubbles from biggest to smallest
    id_sorted = np.argsort(Radius[:,0])
    id_sorted = id_sorted[::-1]
    Radius = Radius[id_sorted]
    Positions = Positions[id_sorted,:]

    def put_in_voxel(Positions, Radius, Voxel_grid):
        voxel_dim = np.asarray(panel_dimension, dtype=np.float32)/np.asarray(grid_shape, dtype=np.float32)
        bubble_ids = np.asarray([i for i in range(number_of_bubbles)], dtype=np.uint32)
        Identifier = np.zeros((number_of_bubbles,3), dtype=np.uint32)

        ## bubble data
        bubbles_data = np.append(Positions - 0.5*e , Radius, axis=1)
        ## add bubbles in voxel grid one at a time, from biggest to smallest
        for bubble_id in bubble_ids:
            for v in product((-1,0,1), repeat=3): # compute ID for each bubble bounding box corner
                direction = np.asarray(v, dtype=np.float32)
                direction = direction/np.linalg.norm(direction)
                ## voxel identifier
                delta_pos = Positions[bubble_id, :] + direction*Radius[bubble_id, :]
                Identifier[bubble_id, 0:2] = np.floor(delta_pos[0:2]*grid_shape[0:2])
                Identifier[bubble_id, 2] = np.floor(((delta_pos[2])/e)*grid_shape[2])
                Identifier[bubble_id, :] = np.clip(Identifier[bubble_id,:], [0,0,0], np.asarray(grid_shape)-1)
                # for each bubble, fill corresponding voxel (if empty)
                if(Voxel_grid[Identifier[bubble_id,0], Identifier[bubble_id,1], Identifier[bubble_id,2], 3] == 0):
                    Voxel_grid[Identifier[bubble_id,0], Identifier[bubble_id,1], Identifier[bubble_id,2], :] = bubbles_data[bubble_id]
                else:
                    print("collide ", Identifier[bubble_id,:])

    ## create and fill voxel grid one bubble at a time
    Voxel_grid = np.zeros(grid_shape+(4,))
    put_in_voxel(Positions, Radius, Voxel_grid)
    return Voxel_grid

def random_bubble_distrib_alexander(grid_shape, radius_range, density, panel_dimension=[1,1,0.01], seed=0):
    rng = np.random.default_rng(seed=seed)
    panel_dimension = np.array(panel_dimension)
    number_of_bubbles = int(density*grid_shape[0]*grid_shape[1]*grid_shape[2])
    #  Position is radius_max away from the border to avoid bubbles outside the grid
    position_min = np.full(3, 0)
    position_max = panel_dimension
    positions = rng.uniform(position_min, position_max, (number_of_bubbles, 3))
    radius = rng.uniform(radius_range[0], radius_range[1], (number_of_bubbles, 1))
    # sort bubbles from biggest to smallest
    idx_sorted = np.argsort(radius, axis=0)[::-1]
    radius = np.take_along_axis(radius, idx_sorted, axis=0)
    positions = np.take_along_axis(positions, idx_sorted, axis=0)

    def insert_one_bubble(position, radius, voxel_grid, idx_min, idx_max):
        index = tuple(slice(idx_min[i], idx_max[i], 1) for i in range(3))
        if np.any(voxel_grid[index][:, :, :, 3] != 0.0):
            return
        voxel_grid[index][:, :, :, :2] = position[:2]  # Position X, Y
        voxel_grid[index][:, :, :, 2] = position[2]  # Position Z
        voxel_grid[index][:, :, :, 3] = radius  # Radius

    idx_min = np.floor((positions - radius) * grid_shape / panel_dimension).astype(np.uint32)
    idx_max = np.ceil((positions + radius) * grid_shape / panel_dimension).astype(np.uint32)
    positions[:, 2] = positions[:, 2] - 0.5 * panel_dimension[2]  # Position Z centered on 0.0
    voxel_grid = np.zeros(grid_shape+(4,))
    for i in range(positions.shape[0]):
        insert_one_bubble(positions[i], radius[i], voxel_grid, idx_min[i], idx_max[i])
    return voxel_grid

# p is a list points [ [x,y,z], ... , [x,y,z] ]
def sdBubble(p: Array3f, r: Float):
    d = dr.norm(p) - r
    return d


# sdf of a filament
# Ax,Ay,Az ,Bx,By,Bz the {start, end} point of each filament section
# Ra, Rb the {start, end} radii of each section
# N the number of sections
def sdBubbles(p: Array3f, bubbles: Array3f, radii: Float, N):
    d = sdBubble(p, Float(0))
    i = UInt32(0)
    N = UInt32(N)
    bubbles = Float(np.ravel(bubbles))
    radii = Float(np.ravel(radii))

    loop = Loop("sdBubbles", lambda: (d,i))
    while loop(i < N):
        bx = Array3f(dr.gather(Array3f,bubbles,i))
        r = Float(dr.gather(Float,Float(radii),i))

        d = dr.minimum(d, sdBubble(p-bx, r))
        i+=1
    return d

def bake_sdf_bubbles(bubbles, panel_dimension, sdf_bake_dimension, MonteCarlo=False):
    X = dr.linspace(Float,0.0,float(panel_dimension[0]), sdf_bake_dimension[0])
    Y = dr.linspace(Float,0.0,float(panel_dimension[1]), sdf_bake_dimension[1])
    Z = dr.linspace(Float,0.0,2*float(panel_dimension[2]), sdf_bake_dimension[2]) - panel_dimension[2]

    X,Y,Z = dr.meshgrid(X,Y,Z)
    #  X,Y= dr.meshgrid(X,Y)
    bubbles = np.reshape(bubbles, (bubbles.shape[0]*bubbles.shape[1]*bubbles.shape[2], 4))

    SDF = sdBubbles(Array3f([X,Y,Z]), bubbles[:,0:3], bubbles[:,3], bubbles.shape[0])

    return SDF

# Add a small bump where bubbles are near the panel surface
def cross_talk_bubble(PosX, PosY, PosZ, grid_shape, bubbles, radius_range, panel_dimension):
    ## usefull variables
    max_dist = ((panel_dimension[2]))*0.01
    D = np.ones_like(PosX, dtype=np.float32)
    R = np.ones_like(PosX, dtype=np.float32)*0

    ## bubble voxel ID from pixel position
    cellX = np.asarray(np.floor(PosX*grid_shape[0]), dtype=np.uint32)
    cellY = np.asarray(np.floor(PosY*grid_shape[1]), dtype=np.uint32)
    cellZ = np.asarray(np.floor(PosZ*grid_shape[2]), dtype=np.uint32)

    ## compute SDF from pixel position to neighboring bubbles
    # look <a> cell around the central cell
    a = 1
    for i in [i for i in range(-a,a+1)]:
        for j in [j for j in range(-a,a+1)]:
            for k in [k for k in range(-a,a+1)]:
                # clip voxel ID
                id_X = np.clip(cellX+i, 0, grid_shape[0]-1)
                id_Y = np.clip(cellY+j, 0, grid_shape[1]-1)
                id_Z = np.clip(cellZ+k, 0, grid_shape[2]-1)

                # bubble at voxel [id_X,id_Y,id_Z] data
                P_X = bubbles[id_X, id_Y, id_Z, 0]
                P_Y = bubbles[id_X, id_Y, id_Z, 1]
                P_Z = bubbles[id_X, id_Y, id_Z, 2]
                R_ = bubbles[id_X, id_Y, id_Z, 3]

                # compute distance
                D_ = np.sqrt((PosX-P_X)**2 + (PosY-P_Y)**2 + (PosZ-P_Z)**2) - R_

                # Union operator
                is_smaller = D < D_
                D = np.minimum(D, D_)
                R = np.where(is_smaller, R_, R)

    ## exponential decay around each bubbles
    # clip voxel ID
    id_X = np.clip(cellX, 0, grid_shape[0]-1)
    id_Y = np.clip(cellY, 0, grid_shape[1]-1)
    id_Z = np.clip(cellZ, 0, grid_shape[2]-1)

    # exponential decay
    sigma = np.clip(R, radius_range[0], radius_range[1])*max_dist
    D = np.exp(-D**2 / sigma)
    D = D / np.max(D)
    D = D*R


    # gaussian smoothing
    sigma = 1
    D = gaussian_filter(D, sigma)

    return D






