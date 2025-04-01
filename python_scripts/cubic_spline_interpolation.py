import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

voxel_size = 0.5
norm = np.sqrt(2)
#x = np.arange(10)
#d = 0.4942573731 
#d = 0.62650329
d=0.75
#x = [-1.5*np.sqrt(2), -1.0*np.sqrt(2), -0.5*np.sqrt(2), 0.0, 0.5*np.sqrt(2), 1.0*np.sqrt(2), 1.5*np.sqrt(2)]
x = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

y = [0.0 , 0.0, (d-0)/(2*voxel_size), (1-0)/(2*voxel_size), (1-d)/(2*voxel_size), 0.0, 0.0]
#y = [0.0 , 0.0, (d-0)/(2*voxel_size*norm), (1-0)/(2*voxel_size*norm), (1-d)/(2*voxel_size*norm), 0.0, 0.0]
#y = [0.0, 0.0, (d-0)/(voxel_size*norm), (1-d)/(voxel_size*norm), 0.0, 0.0, 0.0]
cs = CubicSpline(x, y, bc_type='clamped')
#xs = np.arange(-1.5*np.sqrt(2), 1.5*np.sqrt(2), 0.01)
xs = np.arange(-1.5, 1.5, 0.01)
#ts = np.arange(-1.5*np.sqrt(2), 1.5*np.sqrt(2), 0.0000001)
ts = np.arange(-1.5, 1.5, 0.0000001)


plt.figure()

max_index = np.argmax(cs(ts))
max_x_value = ts[max_index]
plt.axvline(x=max_x_value, color='red', linestyle='--')
plt.text(max_x_value, cs(ts[max_index]), f' {max_x_value/(voxel_size)} diag voxels OR {max_x_value} mm', color='red', verticalalignment='bottom')

#fig, ax = plt.subplots(figsize=(6.5, 4))
plt.plot(x, y,'o')
plt.axhline(y=0.0, color='k')
plt.axvline(x=0.0, color='k')
#ax.plot(xs, np.sin(xs), label='true')
plt.plot(xs, cs(xs))
#ax.plot(xs, cs(xs, 1), label="S'")
#ax.plot(xs, cs(xs, 2), label="S''")
#ax.plot(xs, cs(xs, 3), label="S'''")
#ax.set_xlim(-0.5, 9.5)
#ax.legend(loc='lower left', ncol=2)
plt.title("Cubic spline interpolation, plane, central difference")
plt.show()
