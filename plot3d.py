import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

filename = 'value'

if os.path.exists(filename):
    value = np.load(filename)
    print(value)
    def fun(x, y):
        i = (x + 1.2) *50/1.7
        j = (y + 0.07) *50/0.14
        return value[i][j]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(-1.2, 0.5, 1.7/50.0)
    # y = np.arange(-0.07,0.07,0.14/50.0)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)
    #
    # ax.plot_surface(X, Y, Z)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()
