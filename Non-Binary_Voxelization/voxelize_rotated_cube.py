import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


import math

def euler_xyz_to_axis_angle(euler_angles, normalize=True):
    x_euler, y_euler, z_euler = np.radians(euler_angles)

    # Assuming the angles are in radians.
    c1 = math.cos(y_euler/2)
    s1 = math.sin(y_euler/2)
    c2 = math.cos(z_euler/2)
    s2 = math.sin(z_euler/2)
    c3 = math.cos(x_euler/2)
    s3 = math.sin(x_euler/2)
    c1c2 = c1*c2
    s1s2 = s1*s2
    w = c1c2*c3 - s1s2*s3
    x = c1c2*s3 + s1s2*c3
    y = s1*c2*c3 + c1*s2*s3
    z = c1*s2*c3 - s1*c2*s3
    angle = 2 * math.acos(w)
    if normalize:
        norm = x*x+y*y+z*z
        if norm < 0.001:
            # when all euler angles are zero angle =0 so
            # we can set axis to anything to avoid divide by zero
            x = 1
            y = 0
            z = 0
        else:
            norm = math.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm
    return x, y, z, angle


def write_raw_file(data, filename):
    data.astype(np.float64).tofile(filename)
    print(f"Data written to {filename}")

def write_mhd_file(raw_filename, dimensions, voxel_size, translation, mhd_filename):
    with open(mhd_filename, 'w') as f:
        f.write("ObjectType = Image\n")
        f.write("NDims = 3\n")
        f.write(f"DimSize = {' '.join(map(str, dimensions))}\n")
        f.write(f"ElementSpacing = {' '.join(map(str, voxel_size))}\n")
        f.write(f"Offset = {' '.join(map(str, translation))}\n")
        f.write("ElementType = MET_DOUBLE\n")
        f.write(f"ElementDataFile = {raw_filename}\n")
    print(f"Metadata written to {mhd_filename}")




def rotation_matrix(angles_in_degrees):
    rx, ry, rz = np.radians(angles_in_degrees)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    print(Rz @ Ry @ Rx)
    return Rz @ Ry @ Rx  # Combined rotation matrix

def get_plane_equations(angles_in_degrees, side_length):
    cube_center = [0.0, 0.0, 0.0]
    
    h = side_length / 2

    # Define cube vertices
    vertices = np.array([
        [cube_center[0] - h, cube_center[1] - h, cube_center[2] - h],  # Bottom-left-front
        [cube_center[0] + h, cube_center[1] - h, cube_center[2] - h],  # Bottom-right-front
        [cube_center[0] + h, cube_center[1] + h, cube_center[2] - h],  # Top-right-front
        [cube_center[0] - h, cube_center[1] + h, cube_center[2] - h],  # Top-left-front
        [cube_center[0] - h, cube_center[1] - h, cube_center[2] + h],  # Bottom-left-back
        [cube_center[0] + h, cube_center[1] - h, cube_center[2] + h],  # Bottom-right-back
        [cube_center[0] + h, cube_center[1] + h, cube_center[2] + h],  # Top-right-back
        [cube_center[0] - h, cube_center[1] + h, cube_center[2] + h]   # Top-left-back
    ])

    # Define the six face normals in the unrotated state
    normals = np.array([
        [0, 0, -1],  # Front
        [0, 0, 1],   # Back
        [-1, 0, 0],  # Left
        [1, 0, 0],   # Right
        [0, -1, 0],  # Bottom
        [0, 1, 0]    # Top
    ])

    # Define points that belong to each face
    face_points = np.array([
        vertices[0],  # Front
        vertices[4],  # Back
        vertices[0],  # Left
        vertices[1],  # Right
        vertices[0],  # Bottom
        vertices[3]   # Top
    ])

    # Apply rotation
    R = rotation_matrix(angles_in_degrees)
    rotated_normals = (R @ normals.T).T
    rotated_points = (R @ face_points.T).T

    # Compute plane equations: Ax + By + Cz + D = 0
    planes = []
    for normal, point in zip(rotated_normals, rotated_points):
        A, B, C = normal
        D = -np.dot(normal, point)
        planes.append([A, B, C, D])

    return planes


def three_plane_intersections(a, b, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  
    """

    intersection_points = []

    voxel_planes = []
    for x in [x_min, x_max]:
        voxel_planes.append([1, 0, 0, -x])
    for y in [y_min, y_max]:
        voxel_planes.append([0, 1, 0, -y])
    for z in [z_min, z_max]:
        voxel_planes.append([0, 0, 1, -z])


    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    for voxel_plane in voxel_planes:
        c_vec = np.array(voxel_plane[:3])
        A = np.array([a_vec, b_vec, c_vec])
        d = np.array([-a[3], -b[3], -voxel_plane[3]]).reshape(3,1)

        if np.linalg.det(A) == 0:
            continue

        p_inter = np.linalg.solve(A, d).T
        
        # Check that the intersection point is inside the voxel
        x, y, z = p_inter[0]
        if x >= x_min and x <= x_max and y >= y_min and y <= y_max and z >= z_min and z <= z_max: 
            intersection_points.append(tuple(p_inter[0]))

    return intersection_points

def is_point_inside_rotated_cube(point, plane_equations):
    x, y, z = point

    for plane_equation in plane_equations:
        A, B, C, D = plane_equation
        if (A * x + B * y + C * z + D) * (-D) > 0:
                return False
    return True

def compute_voxel_volume(x_min, x_max, y_min, y_max, z_min, z_max, plane_equations):
    # Check if all the voxel's corners are inside the rotated cube
    
    voxel_totally_inside = True
    voxel_totally_outside = True
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                if not is_point_inside_rotated_cube((x, y, z), plane_equations):
                    voxel_totally_inside = False
                else:
                    voxel_totally_outside = False

    if voxel_totally_inside:
        return voxel_size**3
    if voxel_totally_outside :
        return 0

    # Compute intersection points 
    for i in range(len(plane_equations)):
        for j in range(i+1, len(plane_equations)):
            A1, B1, C1, D1 = plane_equations[i]
            A2, B2, C2, D2 = plane_equations[j]
            if A1 != A2 or B1 != B2 or C1 != C2: # To avoid parallel planes
                intersection_points = three_plane_intersections(plane_equations[i], plane_equations[j], x_min, x_max, y_min, y_max, z_min, z_max)

    # Add voxel vertices that are inside the rotated cube
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                if is_point_inside_rotated_cube((x, y, z), plane_equations):
                    intersection_points.append((x, y, z))

    try:
        hull = ConvexHull(list(set(intersection_points)))
    except QhullError:
        return 0
    return hull.volume


def voxelize_rotated_cube_analytical(voxel_size, grid_size, plane_equations):
    total_volume = 0

    nx, ny, nz = grid_size
    voxel_grid = np.zeros((nx, ny, nz))

    for i in range((-nx/2).astype(int), (nx/2).astype(int)):
        for j in range((-ny/2).astype(int), (ny/2).astype(int)):
            for k in range((-nz/2).astype(int), (nz/2).astype(int)):
                x_min = i * voxel_size
                x_max = (i + 1) * voxel_size
                y_min = j * voxel_size
                y_max = (j + 1) * voxel_size
                z_min = k * voxel_size
                z_max = (k + 1) * voxel_size

                voxel_volume = compute_voxel_volume(x_min, x_max, y_min, y_max, z_min, z_max, plane_equations)
                voxel_grid[i + (nx/2).astype(int), j + (ny/2).astype(int), k + (nz/2).astype(int)] = voxel_volume / voxel_size**3
                total_volume+= voxel_volume

    print("Total volume : ")
    print(total_volume)
    return voxel_grid




side_length = 10.0 # in mm
voxel_size = 0.2

# Ceci semble fonctionner, mais la rotation paraview ne donne pas le même résultat, il faut comprendre pourquoi. 
# De plus, même si on trouve la bonne rotation pour le stl, le code vtk applique une rotation qui empêche de faire le calcul de distance
# /!\ En réalité une rotation autour de x ici équivaut à une rotation en z dans paraview

angles_in_degrees = [45, 0, 0]  # Rotation angles in degrees
#x, y, z, angle = euler_xyz_to_axis_angle(angles_in_degrees)
#print("Angles : ")
#print(np.degrees([x, y, z]))



plane_equations = get_plane_equations(angles_in_degrees, side_length)
print(plane_equations)

grid_size = (side_length/voxel_size*2*np.ones(3)).astype(int) # in voxels

translation_to_origin = (-side_length*2/2 + voxel_size/2)*np.ones(3)

voxel_grid = voxelize_rotated_cube_analytical(voxel_size, grid_size, plane_equations)

raw_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube02.raw"
write_raw_file(voxel_grid, raw_filename)

mhd_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/rotated_cube02.mhd"
write_mhd_file(raw_filename, grid_size, voxel_size*np.ones(3), translation_to_origin, mhd_filename)
