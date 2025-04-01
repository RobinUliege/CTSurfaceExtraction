import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt

def read_raw_file(filename, grid_size):
    arr_1d = np.fromfile(filename, dtype=np.float32)
    return arr_1d.reshape(grid_size).transpose()

def write_raw_file(data, filename):
    data.astype(np.float32).tofile(filename)
    print(f"Data written to {filename}")

def compute_local_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, center):
    x0, y0, z0 = center

    points = []
    # Edges in the X direction:
    for y in [y_min, y_max]:
        for z in [z_min, z_max]:
            x_sq = radius**2 - (y - y0)**2 - (z - z0)**2
            if x_sq < 0:
                #points.append((x_min, y, z))
                #points.append((x_max, y, z))
                pass
            else:
                x_min_sphere = -np.sqrt(x_sq) + x0
                x_max_sphere = np.sqrt(x_sq) + x0
                if x_min_sphere <= x_max and x_max_sphere >= x_min:
                    """print(x_min_sphere)
                    print(x_min)
                    print(x_max_sphere)
                    print(x_max)"""
                    points.append((max(x_min_sphere, x_min), y, z))
                    points.append((min(x_max_sphere, x_max), y, z))


    # Edges in the y direction:
    for x in [x_min, x_max]:
        for z in [z_min, z_max]:
            y_sq = radius**2 - (x - x0)**2 - (z - z0)**2
            if y_sq < 0:
                #points.append((x, y_min, z))
                #points.append((x, y_max, z))
                pass
            else:
                y_min_sphere = -np.sqrt(y_sq) + y0
                y_max_sphere = np.sqrt(y_sq) + y0

                if y_min_sphere <= y_max and y_max_sphere >= y_min:
                    points.append((x, max(y_min_sphere, y_min), z))
                    points.append((x, min(y_max_sphere, y_max), z))

                    """print(y_min_sphere)
                    print(y_min)
                    print(y_max_sphere)
                    print(y_max)"""


    # Edges in  the z direction:
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            z_sq = radius**2 - (x - x0)**2 - (y - y0)**2
            if z_sq < 0:
                #points.append((x, y, z_min))
                #points.append((x, y, z_max))
                pass
            else:
                z_min_sphere = -np.sqrt(z_sq) + z0
                z_max_sphere = np.sqrt(z_sq) + z0

                if z_min_sphere <= z_max and z_max_sphere >= z_min:
                    points.append((x, y, max(z_min_sphere, z_min)))
                    points.append((x, y, min(z_max_sphere, z_max)))
                    """print(z_min_sphere)
                    print(z_min)
                    print(z_max_sphere)
                    print(z_max)"""
        
    #print(list(set(points)))
    #print(len(list(set(points))))
    try:
        hull = ConvexHull(list(set(points)))
    except QhullError:
        return 0
    return hull.volume
    

def compute_voxel_error(density, x_min, x_max, y_min, y_max, z_min, z_max, radius, center, voxel_size):
    x0, y0, z0 = center

    voxel_totally_inside = True
    voxel_totally_outside = True
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                point_distance = (z - z0)**2 + (y - y0)**2 + (x - x0)**2
                if point_distance > radius**2:
                    voxel_totally_inside = False
                if point_distance < radius**2:
                    voxel_totally_outside = False

    if voxel_totally_inside or voxel_totally_outside :
        return 0


    return density*voxel_size**3 - compute_local_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, center)
    #return compute_local_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, center)

def compute_error(data, radius, center, voxel_size, grid_size):
    max_error = 0
    nx, ny, nz = grid_size
    voxel_grid = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_min = i * voxel_size
                x_max = (i + 1) * voxel_size
                y_min = j * voxel_size
                y_max = (j + 1) * voxel_size
                z_min = k * voxel_size
                z_max = (k + 1) * voxel_size

                voxel_grid[i, j, k] = compute_voxel_error(data[i, j, k], x_min, x_max, y_min, y_max, z_min, z_max, radius, center, voxel_size)
                if voxel_grid[i, j, k] > max_error:
                    max_error = voxel_grid[i, j, k]
                    

    return voxel_grid

def plot_error_histogram(data, voxel_size):
    error_on_voxel = np.reshape(voxel_grid, -1)
    error_on_voxel = [error_on_voxel[i] for i in range(len(error_on_voxel)) if error_on_voxel[i] != 0]
    print(error_on_voxel)
    weights = np.ones_like(error_on_voxel)/float(len(error_on_voxel))
    x, bins, p = plt.hist(error_on_voxel, bins = 1000, alpha = 0.45, color = 'blue', weights=weights)
    plt.title('Voxel error, sphere 0.05mm/vox')
    #plt.figtext(0.65, 0.8, "Max err = " + str(round(max(max_error), 6)) + " vox")
    plt.xlabel('Error measurement (mm^3)')
    plt.ylabel('Voxel frequencies')
    plt.savefig('C:/Projets unif/TFE/datafiles/sphere005_error.png')
    plt.show()


radius = 10.0 # in mm
center = np.ones(3)*11
voxel_size = 0.05 # in mm

grid_size = (2*radius/voxel_size*1.2*np.ones(3)).astype(int) # in voxels

data = read_raw_file("C:/Projets unif/TFE/datafiles/sphere005.raw", grid_size)

voxel_grid = compute_error(data, radius, center, voxel_size, grid_size)
write_raw_file(voxel_grid, "C:/Projets unif/TFE/datafiles/sphere005_error.raw")

plot_error_histogram(voxel_grid, voxel_size)

    
