# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from Dataset import Dataset
Gaussian_setting_config = [
    {
        'mean': [0, 0],
        'cov': [[1, 0], [0, 15]],
        'size': 1000
    },
    {
        'mean': [5, -5],
        'cov': [[15, 0], [0, 1]],
        'size': 1000
    }
]
class_prior_config = [0.5, 0.5]
# pause_when_plot = True
# draw_pic = False
draw_pic = True


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


class NonParamMethod:
    def __init__(self):
        pass

    def conditional_probability(self, training_data, x, model_setting):
        pass

    def show_distribution(self, training_data, model_setting):
        x = np.arange(-25, 25, 1, float)
        y = np.arange(-25, 25, 1, float)
        X, Y = np.meshgrid(x, y)
        p = []
        sample_points = np.dstack((X, Y))
        progress = 1
        for row in sample_points:
            p_row = []
            for xi in row:
                p_row.append(self.conditional_probability(training_data, xi, model_setting))
                if progress % 500 == 0:
                    print progress, 'points sampled'
                progress += 1
            p.append(p_row)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap='rainbow')
        plt.pause(.005)
        plt.show()

    def predict(self, x, training_data, model_setting, class_prior):
        return self.conditional_probability(training_data, x, model_setting) * class_prior

    def test(self, all_test_data, all_training_data, model_setting, all_class_prior,show_res=False, max_train=100000):
        # ========== give classify results ============
        results = []
        total_num = 0
        for test_data in all_test_data:
            total_num += len(test_data)
        progress = 1
        for test_data in all_test_data:  # pick one class's test data, index as label
            one_class_result = []
            for item in test_data:
                choose = -1
                p = -1
                for ind, training_data in enumerate(all_training_data):
                    p_tmp = self.predict(item, training_data[:max_train], model_setting, all_class_prior[ind], )
                    if p < p_tmp:
                        p = p_tmp
                        choose = ind
                one_class_result.append(choose)
                if progress % 300 == 0:
                    print progress, "case tested"
                progress += 1
            results.append(one_class_result)
        if show_res:
            print 'predict results:\n'
            for ind, one_class_result in enumerate(results):
                for r in one_class_result:
                    print '', r, ind
        # ======== show recognizing rate ================
        correct = 0.0
        for ind, one_class_result in enumerate(results):
            for epoch in one_class_result:
                if epoch == ind:
                    correct += 1
        s_rate = (correct / total_num)
        print 'success rate:%f model_setting:%f  train_amount %d' % (
            s_rate,
            model_setting,
            len(all_training_data) * min(len(all_training_data[0]), max_train)
        )
        return s_rate


class Parzen(NonParamMethod):
    def __init__(self):
        NonParamMethod.__init__(self)

    def conditional_probability(self, training_data, x, model_setting):
        p = 0
        n = float(len(training_data))
        h1 = model_setting
        hn = 1.0 * h1 / math.sqrt(n)
        for xi in training_data:
            u = np.square((x - xi) / hn)
            tmp_p = 1.0 / (2 * math.pi * hn * hn) * math.exp(-0.5 * np.sum(u))
            p += tmp_p
        return p / n


class TopK(NonParamMethod):
    def __init__(self):
        NonParamMethod.__init__(self)

    def distance_square(self, a, b):
        return np.sum(np.square(a - b))

    def conditional_probability(self, training_data, x, model_setting):
        p = 0
        n = float(len(training_data))
        k1 = model_setting
        kn = int(k1 * math.sqrt(n))
        dst_lst = []
        for xi in training_data:
            dst_lst.append(self.distance_square(xi, x))
        dn_square = sorted(dst_lst)[min(kn, len(dst_lst) - 1)]
        vn = math.pi * dn_square
        return 1.0 * kn / n / vn


def parzen_experiment(data, all_class_test_data):
    pw = Parzen()
    if draw_pic:
        pw.show_distribution(training_data=data.data[0], model_setting=100)
        pw.show_distribution(training_data=data.data[1], model_setting=100)

    # testing
    # fix data amount
    model_setting_lst = [0.01, 0.1, 1, 10, 50, 100, 1000, 10000]
    success_rate_lst = []
    for ms in model_setting_lst:
        success_rate_lst.append(
            pw.test(
                all_test_data=all_class_test_data,
                all_training_data=data.data,
                model_setting=ms,
                all_class_prior=[0.5, 0.5]
            )
        )
    print '====== model setting', model_setting_lst
    print '====== success rate l', success_rate_lst
    # fix window size
    data_amount_lst = [10, 50, 100, 500, 1000]
    success_rate_lst = []
    for da in data_amount_lst:
        success_rate_lst.append(
            pw.test(
                all_test_data=all_class_test_data,
                all_training_data=data.data,
                model_setting=10,
                all_class_prior=[0.5, 0.5],
                max_train=da
            )
        )
    print 'data amount', (np.array(data_amount_lst) * 2).tolist()
    print 'success rate l', success_rate_lst


def topk_experiment(data, all_class_test_data):
    tk = TopK()
    if draw_pic:
        tk.show_distribution(training_data=data.data[0], model_setting=2)
        tk.show_distribution(training_data=data.data[1], model_setting=2)
        # testing
        # fix data amount
    model_setting_lst = [0.01, 0.1, 1, 2, 5, 10, 50, 100, 1000]
    success_rate_lst = []
    for ms in model_setting_lst:
        success_rate_lst.append(
            tk.test(
                all_test_data=all_class_test_data,
                all_training_data=data.data,
                model_setting=ms,
                all_class_prior=[0.5, 0.5]
            )
        )
    print '====== model setting', model_setting_lst
    print '====== success rate l', success_rate_lst
    # fix window size
    data_amount_lst = [10, 50, 100, 500, 1000]
    success_rate_lst = []
    for da in data_amount_lst:
        success_rate_lst.append(
            tk.test(
                all_test_data=all_class_test_data,
                all_training_data=data.data,
                model_setting=10,
                all_class_prior=[0.5, 0.5],
                max_train=da
            )
        )
    print 'data amount', (np.array(data_amount_lst) * 2).tolist()
    print 'success rate l', success_rate_lst


if __name__ == "__main__":
    print "Pattern Recognized Homework 1: Non-Param Method"
    data = Dataset(Gaussian_setting_config)
    all_class_test_data = data.get_test_data([300, 300])
    if draw_pic:
        data.show_data(data.data)
        data.show_data(all_class_test_data)
    print "############## Parzen ################"
    parzen_experiment(data, all_class_test_data)
    print "############## Top K ################"
    topk_experiment(data, all_class_test_data)
    pause()
