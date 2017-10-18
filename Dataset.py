import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class Dataset:
    def __init__(self, Gaussian_setting):
        self.category_count = len(Gaussian_setting)
        self.Gaussian_setting = Gaussian_setting
        self.data = []
        # generate data
        for i in range(self.category_count):
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[i])
            self.data.append(one_class_data)

    def show_data(self, target_data):
        category_count = len(target_data)
        plt_state = [2, category_count, 1]  # plt is param of subplot, [x, y, z] means x row, y col, the Z_th pic
        p_merge = plt.subplot(*[2, category_count, 1 + category_count])
        mark_set = ['r+', 'b+', 'g+', 'k+', 'y+']
        for i in range(category_count):
            p = plt.subplot(*plt_state)
            dimension_split_data = zip(*np.ndarray.tolist(target_data[i]))
            p.plot(dimension_split_data[0], dimension_split_data[1], mark_set[i])
            p_merge.plot(dimension_split_data[0], dimension_split_data[1], mark_set[i])
            plt.pause(.001)  # need when plot in non-block mode
            plt_state[-1] += 1
        # if not pause_when_plot:
        #     plt.ion()
        plt.show()

    def get_test_data(self, size_list):
        # generate data with mixed classes
        test_data = []
        for ind, g_setting in enumerate(self.Gaussian_setting):
            g_setting['size'] = size_list[ind]
            one_class_data = np.random.multivariate_normal(**self.Gaussian_setting[ind])
            test_data.append(one_class_data)
        return np.array(test_data)