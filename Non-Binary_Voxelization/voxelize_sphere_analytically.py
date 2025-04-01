import numpy as np


def compute_voxel_grid(radius, voxel_size, grid_size):
    x_min = 0
    x_max = radius
    y_min = 0
    z_min = 0
    R = radius

    wolf = lambda x: 1/6*( x*y_min*np.sqrt(R**2-y_min**2-x**2)
                          + y_min*(3*R**2+y_min**2)*np.arctan2(x, np.sqrt(R**2-x**2-y_min**2))
                          + (6*x*R**2-2*x**3)*np.arctan2(y_min, np.sqrt(R**2-x**2-y_min**2))
                          - 4*R**3*np.arctan2(x*y_min, R*np.sqrt(R**2-x**2-y_min**2)) )

    integral_part = lambda x: 0.5*( (np.pi/2)*(x*R**2 - (x**3)/3)
                                   - (z_min/2)*(x*np.sqrt(R**2-x**2)+np.arctan2(x, np.sqrt(R**2-x**2))*R**2)
                                   - (y_min/2)*(x*np.sqrt(R**2-x**2-y_min**2)+(R**2-y_min**2)*np.arctan2(x, np.sqrt(R**2-x**2-y_min**2)))
                                   - wolf(x) + x*y_min*z_min )
    
    return integral_part(x_max) - integral_part(x_min)
        

radius = 10.0 # in mm
voxel_size = 0.2

grid_size = (2*radius/voxel_size*1.2*np.ones(3)).astype(int) # in voxels

voxel_grid = compute_voxel_grid(radius, voxel_size, grid_size)
print(voxel_grid/(4/3*np.pi*radius**3/8))

