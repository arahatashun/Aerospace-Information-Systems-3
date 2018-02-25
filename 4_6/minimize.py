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

data = np.array([22.2, 15.1, 15.9, 30.7, 16.2, 41.0,
                 40.3, 35.5, 34.3, 52.3, 42.2, 25.2,
                 17.3, 54.6, 40.4, 32.6, 24.0, 22.3,
                 30.9, 37.0, 21.1, 23.7, 19.7, 21.0,
                 16.4, 34.3, 18.1, 26.8, 12.1, 7.3,
                 15.5, 10.7, 59.4, 30.5, 8.0, 44.0,
                 27.3, 16.0, 12.4, 18.1, 18.1, 34.2,
                 10.3, 33.8, 34.2, 0.8, 32.2, 11.8,
                 26.1, 24.1])


def likelihood(x):
    """ negative log likelihood function

    :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
    :return: log likelihood
    """
    value = -sum([np.log(x[0] * 1 / np.sqrt(2 * np.pi * x[3]) *
                         np.exp(-(data[i] - x[1]) ** 2 / (2 * x[3]))
                         + (1 - x[0]) * 1 / np.sqrt(2 * np.pi * x[4])
                         * np.exp(-(data[i] - x[2]) ** 2 / (2 * x[4])))
                  for i in range(50)])
    return value


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
    plt.savefig("result.pgf")


class EM_algorithm():
    """EM Algorithm for Gaussian Mixture Distribution"""

    def __init__(self, x):
        """Constructor

        :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
        """
        self.x = x
        self.likelihood_list = [likelihood(x)]

    def E_step(self, x):
        """E step calculate responsibility

        :param x: alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
        :return:responsibility_array
        """
        respo = np.array([[(lambda y: x[0] if y == 0 else 1 - x[0])(j)
                           * norm.pdf(x=data[i], loc=x[j + 1], scale=np.sqrt(x[j + 3])) /
                           (x[0] * norm.pdf(x=data[i], loc=x[1], scale=np.sqrt(x[3])) +
                            (1 - x[0]) * norm.pdf(x=data[i], loc=x[2], scale=np.sqrt(x[4])))
                           for i in range(len(data))] for j in range(2)]).T
        return respo

    def M_step(self, x, respo):
        """ Maximum likelihood estimation

        :param x:alpha_f, mu_F, mu_M, sigma^2_F, sigma^2_M
        :param respo:
        :return: x_new
        """
        N_F = sum(respo[:, 0])
        N_M = sum(respo[:, 1])
        mu_F_new = 1 / N_F * sum(respo[:, 0] * data)
        mu_M_new = 1 / N_M * sum(respo[:, 1] * data)
        Sigma_F_new = 1 / N_F * sum(respo[:, 0] * ((data - mu_F_new) ** 2))
        Sigma_M_new = 1 / N_M * sum(respo[:, 1] * ((data - mu_M_new) ** 2))
        pi_F = 1 / len(data) * sum(respo[:, 0])
        x_new = [pi_F, mu_F_new, mu_M_new, Sigma_F_new, Sigma_M_new]
        self.x = x_new

    def iterate(self):
        """ iterate
        :return:negative log likelihood
        """
        respo = self.E_step(self.x)
        self.M_step(self.x, respo)
        likelihood_value = likelihood(self.x)
        self.likelihood_list.append(likelihood_value)
        return likelihood_value

    def plot(self):
        """ plot likelihood change by iteration"""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(u"繰り返し回数")
        ax.set_ylabel(u'負の対数尤度')
        ax.plot(self.likelihood_list)
        print(self.likelihood_list[-1])
        plt.savefig("em.pgf")


def main():
    x = [0.5, 14, 30, 20, 160]
    em = EM_algorithm(x)
    for i in range(1000):
        em.iterate()
    print(em.x)
    plot(em.x)
    em.plot()


if __name__ == '__main__':
    main()
