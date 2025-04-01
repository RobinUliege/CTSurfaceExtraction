import numpy as np

def compute_voxel_portion_in_cube(voxel_grid, cube):
    origin = np.array(voxel_grid['origin'])
    spacing = np.array(voxel_grid['spacing'])
    dimensions = np.array(voxel_grid['dimensions'])

    cube_min = np.array(cube['min_corner'])
    cube_max = np.array(cube['max_corner'])

    portion = np.zeros(dimensions)

    # Sum every voxel volume to check the total volum at the end
    total_volume = 0

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                # Calculate the voxel's min and max corners
                voxel_min = origin + np.array([i, j, k]) * spacing
                voxel_max = voxel_min + spacing

                # Calculate the intersection between the voxel and the cube
                intersection_min = np.maximum(voxel_min, cube_min)
                intersection_max = np.minimum(voxel_max, cube_max)

                # Calculate the volume of the intersection
                intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
                intersection_volume = np.prod(intersection_size)

                voxel_volume = np.prod(spacing)

                portion[i, j, k] = (intersection_volume) / voxel_volume

                # Sum individual volume for checking at the end
                total_volume += intersection_volume

    print("Total volume : ")
    print(total_volume)
    return portion

def write_raw_file(data, filename):
    data.astype(np.float64).tofile(filename)
    print(f"Data written to {filename}")

def write_mhd_file(raw_filename, dimensions, spacing, origin, mhd_filename):
    with open(mhd_filename, 'w') as f:
        f.write("ObjectType = Image\n")
        f.write("NDims = 3\n")
        f.write(f"DimSize = {' '.join(map(str, dimensions))}\n")
        f.write(f"ElementSpacing = {' '.join(map(str, spacing))}\n")
        f.write(f"Offset = {' '.join(map(str, origin))}\n")
        f.write("ElementType = MET_DOUBLE\n")
        f.write(f"ElementDataFile = {raw_filename}\n")
    print(f"Metadata written to {mhd_filename}")


voxel_grid = {
    'origin': [0.0, 0.0, 0.0],
    'spacing': [0.2, 0.2, 0.2],
    'dimensions': [100, 100, 100]
}

cube = {
    'min_corner': [0.75, 0.75, 0.75],
    'max_corner': [10.75, 10.75, 10.75]
}

portion = compute_voxel_portion_in_cube(voxel_grid, cube)

write_raw_file(portion, "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/simple_cube.raw")