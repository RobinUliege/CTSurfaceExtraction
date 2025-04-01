import numpy as np

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)
    
    def __hash__(self):
        return hash(repr(self))

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

def compute_intersection_points(x_min, x_max, y_min, y_max, z_min, z_max, radius, height):
    points = []
    # Edges in the X direction:
    for y in [y_min, y_max]:
        x_sq = radius**2 - y**2
        if x_sq >= 0:
            x_min_sphere = -np.sqrt(x_sq)
            x_max_sphere = np.sqrt(x_sq)
            if x_min_sphere <= x_max and x_max_sphere >= x_min:
                if x_min_sphere >= x_min:
                    points.append(Point(x_min_sphere, y, z_min))
                elif x_max_sphere <= x_max:
                    points.append(Point(x_max_sphere, y, z_min))


    # Edges in the y direction:
    for x in [x_min, x_max]:
        y_sq = radius**2 - x**2
        if y_sq >= 0:
            y_min_sphere = -np.sqrt(y_sq)
            y_max_sphere = np.sqrt(y_sq)

            if y_min_sphere <= y_max and y_max_sphere >= y_min:
                if y_min_sphere >= y_min:
                    points.append(Point(x, y_min_sphere, z_min))
                elif y_max_sphere <= y_max:
                    points.append(Point(x, y_max_sphere, z_min))

    return list(set(points))

def compute_voxel_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, height, voxel_size):
    if z_min >= height/2 or z_max <= -height/2:
        return 0

    voxel_totally_inside = True
    voxel_totally_outside = True
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            point_distance = y**2 + x**2
            if point_distance > radius**2:
                voxel_totally_inside = False
            if point_distance < radius**2:
                voxel_totally_outside = False

    if voxel_totally_inside:
        return voxel_size**3
    if voxel_totally_outside :
        return 0
    
    # First translate the coordinate system to always work in quadrant I (see sketch p. 17)
    if x_min < 0 and y_min >= 0: # Mirror y axis as we are in quadrant II
        tmp = x_min
        x_min = -x_max
        x_max = -tmp
        
    elif x_min < 0 and y_min < 0: # Mirror x and y axes as we are in quadrant III
        tmp = x_min
        x_min = -x_max
        x_max = -tmp
        tmp = y_min
        y_min = -y_max
        y_max = -tmp

    elif x_min >=0 and y_min < 0: # Mirror x axis as we are in quadrant IV
        tmp = y_min
        y_min = -y_max
        y_max = -tmp
        

    # Perform the volume computation as if we were in quadrant I
    intersection_points = compute_intersection_points(x_min, x_max, y_min, y_max, z_min, z_max, radius, height)
    #print("Points : ", intersection_points)

    if intersection_points == []:
        return 0

    highest_point_in_y = None
    highest_point_in_x = None
    for point in intersection_points:
        if highest_point_in_y == None or point.y > highest_point_in_y.y:
            highest_point_in_y = point
        if highest_point_in_x == None or point.x > highest_point_in_x.x:
            highest_point_in_x = point
    
    # If circle intersects with voxels's upper x and/or y bounds, cut the voxel into integrable pieces (see sketch p.17)
    z_lower = z_min
    z_upper = z_max
    R = radius
    cutout_volume = 0

    if highest_point_in_x.x < x_max and highest_point_in_y.y < y_max:
        x_lower = x_min
        x_upper = highest_point_in_x.x
        y_lower = y_min

    elif highest_point_in_x.x < x_max:
        x_lower = highest_point_in_y.x
        x_upper = highest_point_in_x.x
        y_lower = y_min
        cutout_volume = (z_max - z_min) * (y_max - y_min) * (x_lower - x_min)


    elif highest_point_in_y.y < y_max:
        x_lower = x_min
        x_upper = x_max
        y_lower = highest_point_in_x.y
        cutout_volume = (z_max - z_min) * (y_lower - y_min) * (x_max - x_min)


    else:
        x_lower = highest_point_in_y.x
        x_upper = x_max
        y_lower = highest_point_in_x.y
        cutout_volume = (z_max - z_min) * ((y_max-y_min)*(x_lower-x_min) + (y_lower-y_min)*(x_max-x_lower))


    integral_part = lambda x: 0.5*(z_upper-z_lower)*(
        x*np.sqrt(R**2-x**2) + np.arctan2(x, np.sqrt(R**2-x**2))*R**2 - 2*x*y_lower
    )

    voxel_volume = integral_part(x_upper) - integral_part(x_lower) + cutout_volume

    return voxel_volume
        

def generate_voxel_grid(radius, height, voxel_size, grid_size):
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

                voxel_volume = compute_voxel_volume(x_min, x_max, y_min, y_max, z_min, z_max, radius, height, voxel_size)
                voxel_grid[i + (nx/2).astype(int), j + (ny/2).astype(int), k + (nz/2).astype(int)] = voxel_volume / voxel_size**3
                total_volume += voxel_volume

    print(total_volume)
    return voxel_grid




R = 10
y_lower = np.sqrt(100-(5*np.sqrt(2)+0.125)**2)
x_min = np.sqrt(100-(5*np.sqrt(2)+0.125)**2)
x_max = 5*np.sqrt(2) + 0.125
integral_part = lambda x: 0.5*(
        x*np.sqrt(R**2-x**2) + np.arctan2(x, np.sqrt(R**2-x**2))*R**2 - 2*x*y_lower
    )

print(integral_part(x_max)-integral_part(x_min))
exit(0)


radius = 10.0 # in mm
height = 20.0 # in mm
voxel_size = 0.05

grid_size = (2*radius/voxel_size*1.2*np.ones(3)).astype(int) # in voxels

translation_to_origin = (-radius*1.2 + voxel_size/2)*np.ones(3)

voxel_grid = generate_voxel_grid(radius, height, voxel_size, grid_size)

raw_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/cylinder005.raw"
write_raw_file(voxel_grid, raw_filename)

mhd_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/cylinder005.mhd"
write_mhd_file(raw_filename, grid_size, voxel_size*np.ones(3), translation_to_origin, mhd_filename)

