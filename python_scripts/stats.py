import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import subprocess



VOXEL_SIZE = 0.2
TOL = 0.1 # Tolerance is a tenth of a voxel


#algos = ["ExtractSurface", "Poisson", "PowerCrust", "SurfReconst", "SurfaceNets", "FlyingEdges"]
algos = ["ExtractSurfaceIcp"]
#algos = ["ExtractSurface"]
#geometries = ["Circular", "Helical"]
geometries = ["Helical"]

geometries_dict = {
        "Circular": "RekoRolandCropped",
        "Helical": "volRolandHelix"
}

for algo in algos :
    for geometry in geometries : # iterate over geometry_dict instead

        # Generate data
        lines = open(paramFile, 'r').readlines()
        lines[geometryLineIndex] = geometries[algo]
        lines[volumeFilenameIndex] = geometries_dict[algo] # find a way to associate multiple value with the geometry key
        out = open(paramFile, 'w')
        out.writelines(lines)
        out.close()
        subprocess.call(executable + " " + paramFile)


        # Read in the DataFrame
        df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/' + geometry + algo + '.csv')
        #de = pd.read_csv('C:/Projets unif/TFE/errorFiles/' + algo + geometry + 'EdgeErrors.csv')
        
        # plotting histogram 
        weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
        voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
        x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
                alpha = 0.45, color = 'blue', weights=weights)


        plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)

        # display global stats
        plt.figtext(0.65, 0.85, "Global stats :")
        plt.figtext(0.65, 0.8, "Max err = " + str(round(max(voxelRelativeData), 3)) + " vox")
        plt.figtext(0.65, 0.75, "Mean err = " + str(round(np.mean(voxelRelativeData), 3)) + " vox")
        plt.figtext(0.65, 0.7, "std err= " + str(round(np.std(voxelRelativeData), 3)) + " vox")
        values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
        plt.figtext(0.65, 0.65, str(round(len(values_under_tol) / len(df['Distance']) * 100, 1)) + "% tolerated")
        
        # display edge stats
        #plt.figtext(0.65, 0.55, "Edge stats :")
        #plt.figtext(0.65, 0.5, "Max err = " + str(round(de['Max'][0] / VOXEL_SIZE, 3)) + " vox")
        #plt.figtext(0.65, 0.45, "Mean err = " + str(round(de['Mean'][0] / VOXEL_SIZE, 3)) + " vox")
        #plt.figtext(0.65, 0.4, "Stddev err = " + str(round(de['Std'][0] / VOXEL_SIZE, 3)) + " vox")

        
        plt.title(algo + '/' + geometry)
        plt.xlabel('Error measurement (voxels)')
        plt.ylabel('Point frequencies')
        plt.savefig('C:/Projets unif/TFE/graphs/histo' + geometry + algo + '.png')
        plt.show()





# 1a Extract Surface / circular
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/circuExtractSurface.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65,
             str(round(len(values_under_tol) / len(df['Distance']) * 100, 1)) + "% tolerated")
plt.title('Extract Surface / circular')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoCircuExtractSurface.png')
plt.show()

"""

# 1b Extract Surface / helix
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/helixExtractSurface.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('Extract Surface / helical')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoHelixExtractSurface.png')
plt.show()

"""


# 2a Poisson / circular
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/circuPoisson.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('Poisson / circular')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoCircuPoisson.png')
plt.show()

"""


# 2b Poisson / helix
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/helixPoisson.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('Poisson / helical')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoHelixPoisson.png')
plt.show()

"""

# 3a SurfaceReconstruction / circular
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/circuSurfReconst.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('SurfaceReconstruction / circular')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoCircuSurfReconst.png')
plt.show()

"""


# 3b SurfaceReconstruction / helix
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/helixSurfReconst.csv') 
  
# plotting histogram 
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('SurfaceReconstruction / helical')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoHelixSurfReconst.png')
plt.show()

"""


# 4a SurfaceNets/ circular
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/circuSurfaceNets.csv') 
  
# plotting histogram
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('SurfaceNets / circular')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoCircuSurfaceNets.png')
plt.show()

"""


# 4b SurfaceNets/ helix
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/helixSurfaceNets.csv') 
  
# plotting histogram
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('SurfaceNets / helical')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoHelixSurfaceNets.png')
plt.show()

"""


# 5a FlyingEdges / circular
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/circuFlyingEdges.csv') 
  
# plotting histogram
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('FlyingEdges / circular')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoCircuFlyingEdges.png')
plt.show()

"""



# 5b FlyingEdges / helix
  
"""

# Read in the DataFrame 
df = pd.read_csv('C:/Projets unif/TFE/Spreadsheets/helixFlyingEdges.csv') 
  
# plotting histogram
weights = np.ones_like(df['Distance'])/float(len(df['Distance']))
voxelRelativeData = [x / VOXEL_SIZE for x in df['Distance']]
x, bins, p = plt.hist(voxelRelativeData,bins = 100, 
         alpha = 0.45, color = 'blue', weights=weights)


plt.axvspan(TOL, TOL, color = 'red', alpha = 0.5)
plt.figtext(0.55, 0.8, "Max error = " + str(round(max(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.75, "Mean error = " + str(round(np.mean(voxelRelativeData), 6)) + " voxels")
plt.figtext(0.55, 0.7, "std = " + str(round(np.std(voxelRelativeData), 6)) + " voxels")
values_under_tol = [x for x in df['Distance'] if x <= VOXEL_SIZE*TOL]
plt.figtext(0.55, 0.65, "Percent under tolerance = " +
             str(round(len(values_under_tol) / len(df['Distance']), 6)))
plt.title('FlyingEdges / helical')
plt.xlabel('Error measurement (voxels)')
plt.ylabel('Point frequencies')
plt.savefig('C:/Projets unif/TFE/graphs/histoHelixFlyingEdges.png')
plt.show()

"""