import numpy as np
from numpy import array

# export data to .vol file
# 
# we use mitsuba3 .vol file format for RIF saving and loading
# see https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_volumes.html#volume-gridvolume
def export_to_vol(path, data, dimension):
    with open(path, "wb") as file:
        print("export RIF to binary")

        file.write("I".encode())
        file.write("O".encode())
        file.write("R".encode())

    with open(path, "ab") as file:
        version = 3;
        A = [version, 1, dimension[0], dimension[1], dimension[2], 1]

        file.write((A[0]).to_bytes(1, byteorder='little', signed=False))
        for i  in range(1,6):
            file.write((A[i]).to_bytes(4, byteorder='little', signed=False))


    with open(path, "ab") as file:
        # not used
        xmin = 0
        ymin = 0
        zmin = 0
        xmax = 0
        ymax = 0
        zmax = 0
        A = np.asarray([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)
        A = np.ravel(A)
        A = array(A, 'float32')
        A.tofile(file)


    with open(path, "ab") as file:
        print("writing data...")
        A = array(data, 'float32',order='F')
        A.tofile(file)
        
        return True
    print("writing failed at path ", path)
    return False

# export data to .bubble file
# 
# we use mitsuba3 .vol file format for RIF saving and loading the bubble field
# each voxel stores 4 channels: position.xyz and radius
# see https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_volumes.html#volume-gridvolume
def export_to_bubble(path, data, dimension):
    with open(path, "wb") as file:
        print("export Bubbles to binary")

        file.write("I".encode())
        file.write("O".encode())
        file.write("R".encode())

    with open(path, "ab") as file:
        version = 3;
        A = [version, 1, dimension[0], dimension[1], dimension[2], 4]

        file.write((A[0]).to_bytes(1, byteorder='little', signed=False))
        for i  in range(1,6):
            file.write((A[i]).to_bytes(4, byteorder='little', signed=False))


    with open(path, "ab") as file:
        # not used
        xmin = 0
        ymin = 0
        zmin = 0
        xmax = 0
        ymax = 0
        zmax = 0
        A = np.asarray([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)
        A = np.ravel(A)
        A = array(A, 'float32')
        A.tofile(file)


    with open(path, "ab") as file:

        # switch X and Y axis for mitsuba3 coordinates ordering
        data_X = np.copy(data[:,:,:,0] )
        data_Y = np.copy(data[:,:,:,1] )
        data_Z = np.copy(data[:,:,:,2] )
        data_R = np.copy(data[:,:,:,3] )
        data[:,:,:,0] = data_Y
        data[:,:,:,1] = data_X
        data[:,:,:,2] = data_Z
        data[:,:,:,3] = data_R

        A = array(data, 'float32',order='C')
        A = np.ravel(A)
        A.tofile(file)
        
        return True
    print("writing failed at path ", path)
    return False
