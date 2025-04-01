import numpy as np
from scipy.spatial import ConvexHull

def compute_voxel_portion_in_rotated_cube(voxel_grid, cube):
    """
    Computes the portion of each voxel's volume that is enclosed within a rotated cube using ConvexHull.

    Parameters:
    voxel_grid (dict): A dictionary containing the voxel grid information:
                       - 'origin': The origin of the voxel grid (x, y, z).
                       - 'spacing': The spacing between voxels (dx, dy, dz).
                       - 'dimensions': The dimensions of the voxel grid (nx, ny, nz).
    cube (dict): A dictionary containing the cube information:
                 - 'center': The center of the cube (x_center, y_center, z_center).
                 - 'size': The size of the cube (width, height, depth).
                 - 'rotation': The rotation angles in degrees (rx, ry, rz).

    Returns:
    np.ndarray: A 3D array where each element represents the portion of the corresponding voxel's volume that is enclosed within the rotated cube.
    """
    # Extract voxel grid information
    origin = np.array(voxel_grid['origin'])
    spacing = np.array(voxel_grid['spacing'])
    dimensions = np.array(voxel_grid['dimensions'])
    resolution = voxel_grid['spacing'][0] / 10

    # Extract cube information
    cube_center = np.array(cube['center'])
    cube_size = np.array(cube['size'])
    rotation_angles = np.radians(np.array(cube['rotation']))  # Convert degrees to radians

    # Initialize the result array
    portion = np.zeros(dimensions)

    # Compute the rotation matrix
    rx, ry, rz = rotation_angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx  # Combined rotation matrix

    # Define the cube's vertices in its local coordinate system
    cube_half_size = cube_size / 2
    cube_vertices_local = np.array([
        [-cube_half_size[0], -cube_half_size[1], -cube_half_size[2]],
        [ cube_half_size[0], -cube_half_size[1], -cube_half_size[2]],
        [-cube_half_size[0],  cube_half_size[1], -cube_half_size[2]],
        [ cube_half_size[0],  cube_half_size[1], -cube_half_size[2]],
        [-cube_half_size[0], -cube_half_size[1],  cube_half_size[2]],
        [ cube_half_size[0], -cube_half_size[1],  cube_half_size[2]],
        [-cube_half_size[0],  cube_half_size[1],  cube_half_size[2]],
        [ cube_half_size[0],  cube_half_size[1],  cube_half_size[2]],
    ])

    # Transform the cube's vertices to world coordinates
    cube_vertices_world = (rotation_matrix @ cube_vertices_local.T).T + cube_center

    total_volume = 0

    # Iterate over each voxel
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                # Calculate the voxel's vertices in world coordinates
                voxel_min = origin + np.array([i, j, k]) * spacing
                voxel_max = voxel_min + spacing
                voxel_vertices = np.array([
                    [voxel_min[0], voxel_min[1], voxel_min[2]],
                    [voxel_max[0], voxel_min[1], voxel_min[2]],
                    [voxel_min[0], voxel_max[1], voxel_min[2]],
                    [voxel_max[0], voxel_max[1], voxel_min[2]],
                    [voxel_min[0], voxel_min[1], voxel_max[2]],
                    [voxel_max[0], voxel_min[1], voxel_max[2]],
                    [voxel_min[0], voxel_max[1], voxel_max[2]],
                    [voxel_max[0], voxel_max[1], voxel_max[2]],
                ])

                """ This is the section that needs to be reworked
                # Find the intersection points between the voxel and the cube
                intersection_points = []
                for v in voxel_vertices:
                    if np.all(np.abs(rotation_matrix.T @ (v - cube_center)) <= cube_half_size):
                        intersection_points.append(v)
                for c in cube_vertices_world:
                    if np.all((c >= voxel_min) & (c <= voxel_max)):
                        intersection_points.append(c)

                """
                intersection_points = []
                for l in range(10):
                    for m in range(10):
                        for n in range(10):
                            #if [l, m, n] in 
                            continue
                
                
                # Compute the intersection volume using ConvexHull
                if len(intersection_points) >= 4:
                    try:
                        hull = ConvexHull(intersection_points)
                        intersection_volume = hull.volume
                    except:
                        intersection_volume = 0
                    
                else:
                    intersection_volume = 0.0

                voxel_volume = np.prod(spacing)

                # Compute the portion of the voxel's volume that is enclosed within the cube
                portion[i, j, k] = intersection_volume / voxel_volume
                total_volume += intersection_volume

    print('Total volume : ')
    print(total_volume)
    return portion

def write_raw_file(data, filename):
    """
    Writes a 3D numpy array to a raw file in binary format.

    Parameters:
    data (np.ndarray): The 3D array to be saved.
    filename (str): The name of the output raw file.
    """
    data.astype(np.float32).tofile(filename)
    print(f"Data written to {filename}")

def write_mhd_file(raw_filename, dimensions, spacing, origin, mhd_filename):
    """
    Writes a metadata file in .mhd format.

    Parameters:
    raw_filename (str): The name of the raw file containing the data.
    dimensions (tuple): The dimensions of the volume (nx, ny, nz).
    spacing (tuple): The spacing between voxels (dx, dy, dz).
    origin (tuple): The origin of the volume (x, y, z).
    mhd_filename (str): The name of the output .mhd file.
    """
    with open(mhd_filename, 'w') as f:
        f.write("ObjectType = Image\n")
        f.write("NDims = 3\n")
        f.write(f"DimSize = {' '.join(map(str, dimensions))}\n")
        f.write(f"ElementSpacing = {' '.join(map(str, spacing))}\n")
        f.write(f"Offset = {' '.join(map(str, origin))}\n")
        f.write("ElementType = MET_FLOAT\n")
        f.write(f"ElementDataFile = {raw_filename}\n")
    print(f"Metadata written to {mhd_filename}")



# Main
points = [[0, 0, 0], [0, 2, 0], [0, 0, 2], [0, 2, 2], [2, 0, 0], [2, 2, 0], [2, 0, 2], [2, 2, 2]]
hull = ConvexHull(points)
print(hull.volume)



voxel_grid = {
    'origin': [-10.0, -10.0, -10.0],
    'spacing': [1.0, 1.0, 1.0],
    'dimensions': [100, 100, 100]
}

cube = {
    'center': [0.5, 0.5, 0.5],
    'size': [5.0, 5.0, 5.0],
    'rotation': [0, 0, 0]  # Rotation angles in degrees (rx, ry, rz)
}

# Compute the portion of each voxel's volume inside the rotated cube
portion = compute_voxel_portion_in_rotated_cube(voxel_grid, cube)

# Write the result to a raw file
raw_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube.raw"
write_raw_file(portion, raw_filename)

# Write the corresponding .mhd metadata file
mhd_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube.mhd"
write_mhd_file(raw_filename, voxel_grid['dimensions'], voxel_grid['spacing'], voxel_grid['origin'], mhd_filename)