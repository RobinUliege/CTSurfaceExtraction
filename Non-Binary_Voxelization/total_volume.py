import numpy as np

voxel_size = 0.05
radius = 10.0
height = 20

true_volume = np.pi*radius**2*height

voxelization_filename = "C:/Projets unif/TFE/Non-Binary_Voxelization/Output/cylinder005.raw"

arr_1d = np.fromfile(voxelization_filename, dtype=np.float64)
print(np.sum(arr_1d * voxel_size**3))
print("True volume : ")
print(true_volume)