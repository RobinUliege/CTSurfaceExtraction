import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import tplquad

def write_raw_file(data, filename):
    data.astype(np.float32).tofile(filename)
    print(f"Data written to {filename}")

def write_mhd_file(raw_filename, dimensions, spacing, origin, mhd_filename):
    with open(mhd_filename, 'w') as f:
        f.write("ObjectType = Image\n")
        f.write("NDims = 3\n")
        f.write(f"DimSize = {' '.join(map(str, dimensions))}\n")
        f.write(f"ElementSpacing = {' '.join(map(str, spacing))}\n")
        f.write(f"Offset = {' '.join(map(str, origin))}\n")
        f.write("ElementType = MET_FLOAT\n")
        f.write(f"ElementDataFile = {raw_filename}\n")
    print(f"Metadata written to {mhd_filename}")

"""
def triple_integral(x_min, x_max, y_min, y_max, z_min, z_max, radius, center):
    f = lambda x, y, z: 1
    z = lambda x, y: radius**2 - (x - x_min)**2 - (y - y_min)**2
    return tplquad(f, x_min, x_max, lambda y: y_min, lambda y: y_max, -z, z)[0]
"""


def sphere_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, center):
    x0, y0, z0 = center

    # Check if all the voxel's corners are inside the sphere
    
    voxel_totally_inside = True
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                if (z - z0)**2 + (y - y0)**2 + (x - x0)**2 > radius**2:
                    voxel_totally_inside = False
                    
    if voxel_totally_inside:
        return voxel_size**3
    

        
    def integrand(y, x):
        # Check if point (x, y, z) is inside the sphere
        z_sq = radius**2 - (x - x0)**2 - (y - y0)**2
        if z_sq < 0:
            return 0.0
        z_max_sphere = np.sqrt(z_sq) + z0
        z_min_sphere = -np.sqrt(z_sq) + z0

        # Clip the z bounds to the voxel's z bounds
        z_lower = max(z_min_sphere, z_min)
        z_upper = min(z_max_sphere, z_max)

        if z_lower >= z_upper:
            return 0.0
        return z_upper - z_lower
    

    y_min_bound = lambda x: max(y_min, -np.sqrt(radius**2 - (x - x0)**2) + y0)
    y_max_bound = lambda x: min(y_max, np.sqrt(radius**2 - (x - x0)**2) + y0)


    # Integrate over x and y
    #volume, _ = dblquad(integrand, x_min, x_max, y_min, y_max)
    volume, _ = dblquad(integrand, x_min, x_max, y_min_bound, y_max_bound)
    return volume


def voxelize_sphere_analytical(radius, center, voxel_size, grid_size):
    total_volume = 0
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

                voxel_grid[i, j, k] = sphere_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, center) / voxel_size**3
                total_volume += voxel_grid[i, j, k] * voxel_size**3

    print(total_volume)
    return voxel_grid


radius = 10.0 # in mm
center = (11, 11, 11)
voxel_size = 0.2

grid_size = (2*radius/voxel_size*1.2*np.ones(3)).astype(int) # in voxels
print(grid_size)
#grid_size = (10, 10, 10)

voxel_grid = voxelize_sphere_analytical(radius, center, voxel_size, grid_size)

raw_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/sphere02.raw"
write_raw_file(voxel_grid, raw_filename)

mhd_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/sphere.mhd"
#write_mhd_file(mhd_filename, (grid_size*voxel_size).astype(int), [voxel_size, voxels_size, voxel_size], [0, 0, 0]], mhd_filename)