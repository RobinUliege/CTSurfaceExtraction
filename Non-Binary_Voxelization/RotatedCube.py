import numpy as np
from scipy.spatial import ConvexHull

def compute_voxel_portion_in_rotated_cube(voxel_grid, cube):
    """
    Computes the portion of each voxel's volume that is enclosed within a rotated cube.

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
    origin = np.array(voxel_grid['origin'])
    spacing = np.array(voxel_grid['spacing'])
    dimensions = np.array(voxel_grid['dimensions'])

    cube_center = np.array(cube['center'])
    cube_size = np.array(cube['size'])
    rotation_angles = np.radians(np.array(cube['rotation']))  # Convert degrees to radians

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

    # Sum every voxel volume to check the total volum at the end
    total_volume = 0

    # Iterate over each voxel
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                # Calculate the voxel's center in world coordinates
                voxel_center = origin + np.array([i, j, k]) * spacing

                # Transform the voxel center into the cube's local coordinate system
                voxel_local = rotation_matrix.T @ (voxel_center - cube_center)

                # Check if the voxel center is inside the cube
                #if np.all(np.abs(voxel_local) <= cube_size / 2):
                if True:
                    # If inside, compute the intersection volume
                    intersection_min = np.maximum(voxel_local - spacing / 2, -cube_size / 2)
                    intersection_max = np.minimum(voxel_local + spacing / 2, cube_size / 2)
                    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
                    intersection_volume = np.prod(intersection_size)

                    # Calculate the voxel's volume
                    voxel_volume = np.prod(spacing)

                    # Compute the portion of the voxel's volume that is enclosed within the cube
                    portion[i, j, k] = intersection_volume / voxel_volume

                    # Sum individual volume for checking at the end
                    total_volume += intersection_volume / voxel_volume

    print("Total volume : ")
    print(total_volume)
    return portion

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

voxel_grid = {
    'origin': [-10.0, -10.0, -10.0],
    'spacing': [0.2, 0.2, 0.2],
    'dimensions': [100, 100, 100]
}

cube = {
    'center': [0.0, 0.0, 0.0],
    'size': [5.0, 5.0, 5.0],
    'rotation': [45, 0, 0]  # Rotation angles in degrees (rx, ry, rz)
}

portion = compute_voxel_portion_in_rotated_cube(voxel_grid, cube)

raw_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube.raw"
write_raw_file(portion, raw_filename)

mhd_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube.mhd"
write_mhd_file(raw_filename, voxel_grid['dimensions'], voxel_grid['spacing'], voxel_grid['origin'], mhd_filename)
