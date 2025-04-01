import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import cv2 as cv
from scipy.misc import derivative
from scipy.ndimage import center_of_mass
from test import *


# Functions to create the circle
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

def compute_intersection_points(x_min, x_max, y_min, y_max, radius):
    points = []
    # Edges in the X direction:
    for y in [y_min, y_max]:
        x_sq = radius**2 - y**2
        if x_sq >= 0:
            x_min_sphere = -np.sqrt(x_sq)
            x_max_sphere = np.sqrt(x_sq)
            if x_min_sphere <= x_max and x_max_sphere >= x_min:
                if x_min_sphere >= x_min:
                    points.append(Point(x_min_sphere, y, 0))
                elif x_max_sphere <= x_max:
                    points.append(Point(x_max_sphere, y, 0))


    # Edges in the y direction:
    for x in [x_min, x_max]:
        y_sq = radius**2 - x**2
        if y_sq >= 0:
            y_min_sphere = -np.sqrt(y_sq)
            y_max_sphere = np.sqrt(y_sq)

            if y_min_sphere <= y_max and y_max_sphere >= y_min:
                if y_min_sphere >= y_min:
                    points.append(Point(x, y_min_sphere, 0))
                elif y_max_sphere <= y_max:
                    points.append(Point(x, y_max_sphere, 0))

    return list(set(points))

def compute_voxel_volume(x_min, x_max, y_min, y_max, radius, voxel_size):

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
    intersection_points = compute_intersection_points(x_min, x_max, y_min, y_max, radius)
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
        cutout_volume = (y_max - y_min) * (x_lower - x_min)


    elif highest_point_in_y.y < y_max:
        x_lower = x_min
        x_upper = x_max
        y_lower = highest_point_in_x.y
        cutout_volume = (y_lower - y_min) * (x_max - x_min)


    else:
        x_lower = highest_point_in_y.x
        x_upper = x_max
        y_lower = highest_point_in_x.y
        cutout_volume = ((y_max-y_min)*(x_lower-x_min) + (y_lower-y_min)*(x_max-x_lower))


    integral_part = lambda x: 0.5*(
        x*np.sqrt(R**2-x**2) + np.arctan2(x, np.sqrt(R**2-x**2))*R**2 - 2*x*y_lower
    )

    voxel_volume = integral_part(x_upper) - integral_part(x_lower) + cutout_volume

    return voxel_volume
        

def generate_voxel_grid(radius, voxel_size, grid_size):
    total_volume = 0

    nx, ny = grid_size
    voxel_grid = np.zeros((nx, ny))

    for i in range((-nx/2).astype(int), (nx/2).astype(int)):
        for j in range((-ny/2).astype(int), (ny/2).astype(int)):
            x_min = i * voxel_size
            x_max = (i + 1) * voxel_size
            y_min = j * voxel_size
            y_max = (j + 1) * voxel_size

            voxel_volume = compute_voxel_volume(x_min, x_max, y_min, y_max, radius, voxel_size)
            voxel_grid[i + (nx/2).astype(int), j + (ny/2).astype(int)] = voxel_volume / voxel_size**3
            total_volume += voxel_volume

    return voxel_grid

# functions to create square
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
            # Calculate the voxel's min and max corners
            voxel_min = origin + np.array([i, j]) * spacing
            voxel_max = voxel_min + spacing

            # Calculate the intersection between the voxel and the cube
            intersection_min = np.maximum(voxel_min, cube_min)
            intersection_max = np.minimum(voxel_max, cube_max)

            # Calculate the volume of the intersection
            intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
            intersection_volume = np.prod(intersection_size)

            voxel_volume = np.prod(spacing)

            portion[i, j] = (intersection_volume) / voxel_volume

            # Sum individual volume for checking at the end
            total_volume += intersection_volume

    #print("Total volume : ")
    #print(total_volume)
    return portion


# Main
# Parameters for square
voxel_grid = {
    'origin': [0.0, 0.0],
    'spacing': [1.0, 1.0],
    'dimensions': [100, 100]
}

cube = {
    'min_corner': [20.125, 20.125],
    'max_corner': [80.125, 80.125]
}

square_img = compute_voxel_portion_in_cube(voxel_grid, cube)

# Parameters for circle
radius = 30.0 # in mm
voxel_size = 1

grid_size = (100*np.ones(2)).astype(int) # in voxels
circle_img = generate_voxel_grid(radius, voxel_size, grid_size)


# Algo parameters
kernel_size = 3
max_interp_window_size = 3
pixel_to_interpolate = np.array([20, 20]) # [20, 20], [20, 50], [29, 28]
alpha_deriche = 5
boundary_condition = 'clamped'
normal_dir = np.array([1, 1])
noise_std = 0.00
plot_results = True
img = square_img # circle_img or square_img
 
# Add noise to the image
gaussian_noise = np.random.normal(0, noise_std, size=voxel_grid['dimensions'])
img = img + gaussian_noise

# Gradient computation
dir_magnitude = np.linalg.norm(normal_dir)

 # Central difference
central_difference_array = np.zeros(voxel_grid['dimensions'])
for i in range(1,voxel_grid['dimensions'][0]-1):
    for j in range(1, voxel_grid['dimensions'][1]-1):
        h_comp = normal_dir[0]
        v_comp = normal_dir[1]
        nextVox = img[i+h_comp][j+v_comp]
        prevVox = img[i-h_comp][j-v_comp]
        #central_difference_array[i][j] = (img[i+1][j+1] - img[i-1][j-1]) / (2*voxel_grid['spacing'][0]*dir_magnitude)
        central_difference_array[i][j] = (nextVox - prevVox) / (2*voxel_grid['spacing'][0]*dir_magnitude)


 # Sobel
#laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=kernel_size)/6.0
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=kernel_size)/6.0

sobel_magnitude = np.sqrt((sobelx ** 2) + (sobely ** 2))
orientation = np.arctan2(sobely, sobelx) * (180 / np.pi) % 180

 # Deriche
#cdArr = np.array([0.25,0.5,0.75,1,2,3,4,5,10])#float(sys.argv[1])
cdArr = np.array([alpha_deriche])
#cdArr = np.array([3])#float(sys.argv[1])
allCoordsCG = np.zeros(cdArr.size)
allCoordsInt = np.zeros(cdArr.size)
k = 0
for cd in cdArr:
    #Deriche gradient map
    gColsP = dericheColsPlus(img,cd)
    gColsM = dericheColsMinus(img,cd)
    thetaCols = -(1-exp(-cd))**2 * (gColsP+gColsM)

    gRowsP = dericheRowsPlus(img,cd)
    gRowsM = dericheRowsMinus(img,cd)
    thetaRows = -(1-exp(-cd))**2 * (gRowsP+gRowsM)

    theta = np.abs(thetaRows) + np.abs(thetaCols)
    window = 1 # was 11
    #directional derivative in Deriche
    deriche_magnitude = np.zeros(voxel_grid['dimensions'])
    for i in range(1,voxel_grid['dimensions'][0]-1):
        for j in range(1, voxel_grid['dimensions'][1]-1):
            #pt = np.array([10,41])
            pt = np.array([i, j])
            ind = np.unravel_index(np.argmax(theta, axis=None), theta.shape)
            #pt = np.array([ind[0], ind[1]])
            v  = np.array([1,1])
            gradientV = directionalDerivative(theta,pt,v,window)
            deriche_magnitude[i][j] = gradientV[0]
            



# Compute interpolation and center of gravity at a given pixel

for interp_window_size in range(3, max_interp_window_size+1):
    x_inter = np.zeros(2*interp_window_size + 1)
    y_central_difference = np.zeros(2*interp_window_size + 1)
    y_sobel = np.zeros(2*interp_window_size + 1)
    y_deriche = np.zeros(2*interp_window_size + 1)
    i, j = pixel_to_interpolate

    for l in range(-interp_window_size, interp_window_size+1):
        x_position = l*voxel_grid['spacing'][0]*dir_magnitude
        x_inter[l + interp_window_size] = x_position

        pxIndex = [i + l * normal_dir[0], j + l * normal_dir[1]]
        y_central_difference[l + interp_window_size] = central_difference_array[pxIndex[0]][pxIndex[1]]

        y_sobel[l + interp_window_size] = sobel_magnitude[pxIndex[0]][pxIndex[1]]

        y_deriche[l + interp_window_size] = deriche_magnitude[pxIndex[0]][pxIndex[1]]


    center_of_gravity_sobel = center_of_mass(y_sobel)[0]*dir_magnitude - interp_window_size*dir_magnitude
    center_of_gravity_deriche = center_of_mass(y_deriche)[0]*dir_magnitude - interp_window_size*dir_magnitude
    center_of_gravity_central_difference = center_of_mass(y_central_difference)[0]*dir_magnitude - interp_window_size*dir_magnitude



    ts = np.arange(-interp_window_size*dir_magnitude, interp_window_size*dir_magnitude, 0.000001)

    cs_central = CubicSpline(x_inter, y_central_difference, bc_type=boundary_condition)
    max_index = np.argmax(abs(cs_central(ts)))
    max_x_value = ts[max_index]
    plt.axvline(x=max_x_value, color='purple', linestyle='--')
    plt.text(max_x_value, cs_central(ts[max_index]), f'{max_x_value}', color='purple', verticalalignment='bottom')

    cs_sobel = CubicSpline(x_inter, y_sobel, bc_type=boundary_condition)
    max_index = np.argmax(abs(cs_sobel(ts)))
    max_x_value = ts[max_index]
    plt.axvline(x=max_x_value, color='orange', linestyle='--')
    plt.text(max_x_value, cs_sobel(ts[max_index]), f'{max_x_value}', color='orange', verticalalignment='top')

    cs_deriche = CubicSpline(x_inter, y_deriche, bc_type=boundary_condition)
    max_index = np.argmax(abs(cs_deriche(ts)))
    max_x_value = ts[max_index]
    plt.axvline(x=max_x_value, color='green', linestyle='--')
    plt.text(max_x_value, cs_deriche(ts[max_index]), f'{max_x_value}', color='green', verticalalignment='top')

    xs = np.arange(-interp_window_size*dir_magnitude, interp_window_size*dir_magnitude, 0.0001)
    plt.plot(xs, cs_central(xs))
    plt.plot(xs, cs_sobel(xs))
    plt.plot(xs, cs_deriche(xs))


    print(f"center of gravity for Sobel : {center_of_gravity_sobel}")
    print(f"center of gravity for Deriche : {center_of_gravity_deriche}")


    for h in range(-interp_window_size, interp_window_size+1):
        plt.axvline(x=h*dir_magnitude, color='red', linestyle='-')
     
    x_title=cube['min_corner'][0]
    y_title=cube['min_corner'][1]
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(["max_offset_central_diff", "max_offset_sobel", "max_offset_deriche", "Central Difference", "Sobel", "Deriche"])
    plt.figtext(0.75, 0.65, f"Center of gravity Central Diff : {round(center_of_gravity_central_difference, 6)}")
    plt.figtext(0.75, 0.6, f"Center of gravity Sobel : {round(center_of_gravity_sobel, 6)}")
    plt.figtext(0.75, 0.55, f"Center of gravity Deriche : {round(center_of_gravity_deriche, 6)}")
    plt.figtext(0.75, 0.5, f"True surface position : {round((0.5 - img[tuple(pixel_to_interpolate)])*dir_magnitude, 6)}") # Must be changed depending on the position of the surface relative to the center of the voxel

    plt.grid()
    plt.title(f'cubic spline interpolation on square (x={x_title}, y={y_title}), alpha_deriche={alpha_deriche}, interp_window={interp_window_size}')


# Plot results
if(plot_results):

    plt.figure()
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
    #plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
    #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,4),plt.imshow(sobelxy,cmap = 'gray')
    #plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(central_difference_array,cmap = 'gray')
    plt.title('central difference'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobel_magnitude,cmap = 'gray')
    plt.title('Sobel gradient magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(deriche_magnitude,cmap = 'gray')
    plt.title('Deriche gradient magnitude'), plt.xticks([]), plt.yticks([])
    

    plt.show()


