#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for problem 4-6
"""
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'IPAPGothic'
import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution

data = [22.2, 15.1, 15.9, 30.7, 16.2, 41.0,
        40.3, 35.5, 34.3, 52.3, 42.2, 25.2,
        17.3, 54.6, 40.4, 32.6, 24.0, 22.3,
        30.9, 37.0, 21.1, 23.7, 19.7, 21.0,
        16.4, 34.3, 18.1, 26.8, 12.1, 7.3,
        15.5, 10.7, 59.4, 30.5, 8.0, 44.0,
        27.3, 16.0, 12.4, 18.1, 18.1, 34.2,
        10.3, 33.8, 34.2, 0.8, 32.2, 11.8,
        26.1, 24.1]


def likelihood(x):
    """ negative log likelihood function

    :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
    :return: log likelihood
    """
    return -sum([np.log(x[0] * 1 / np.sqrt(2 * np.pi * x[3]) * np.exp(-(data[i] - x[1]) ** 2 / (2 * x[3])) \
                        + (1 - x[0]) * 1 / np.sqrt(2 * np.pi * x[4]) * np.exp(-(data[i] - x[2]) ** 2 / (2 * x[4])))
                 for i in range(50)])


def optimize_diff_ev():
    """ minimize negative log likelihood using differntial evolution

    :return:array of x
    """
    bounds = [(0, 1), (0, 50), (0, 50), (0, 200), (0, 200)]
    result = differential_evolution(likelihood, bounds)
    print(result.fun)
    print(result.x)
    return result.x


def gaussian_mixture_distribution(meter, x):
    """

    :param meter:
    :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
    :return: 
    """
    return [x[0] * norm.pdf(x=meter, loc=x[1], scale=np.sqrt(x[3])) \
            + (1 - x[0]) * norm.pdf(x=meter, loc=x[2], scale=np.sqrt(x[4])),
            x[0] * norm.pdf(x=meter, loc=x[1], scale=np.sqrt(x[3])),
            (1 - x[0]) * norm.pdf(x=meter, loc=x[2], scale=np.sqrt(x[4]))]


def plot(x):
    """make figure
    :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(u"記録 m")
    ax.set_ylabel(u'人数')
    ax.hist(data, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    meter_list = np.linspace(0, 60, 100)
    ax.plot(meter_list, 250 * gaussian_mixture_distribution(meter_list, x)[0], label=u'混合ガウス分布')
    ax.plot(meter_list, 250 * gaussian_mixture_distribution(meter_list, x)[1], label=u"female")
    ax.plot(meter_list, 250 * gaussian_mixture_distribution(meter_list, x)[2], label=u"male")
    ax.legend(loc="upper left")
    plt.savefig("evolution.pgf")


def main():
    x = optimize_diff_ev()
    plot(x)


if __name__ == '__main__':
    main()
