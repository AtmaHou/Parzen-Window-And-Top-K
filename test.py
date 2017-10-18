from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 1)
Y = np.arange(-4, 4, 1)
X, Y = np.meshgrid(X, Y)
xxx =  np.dstack((X,Y))
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='rainbow')

plt.show()
print xxx

# fig = plt.figure()
# ax = Axes3D(fig)
# x = np.arange(-50, 50, 0.1, float)
# y = np.arange(-50, 50, 0.1, float)
# X,Y = np.meshgrid(x,y)
# Z = X + Y
# ax.plot_surface(X, Y, Z)
# plt.show()