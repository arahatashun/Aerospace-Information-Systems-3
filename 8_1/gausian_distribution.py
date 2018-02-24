#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for problem 8-1
"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'IPAPGothic'
import numpy as np


def multi_gausian(mean, sigma):
    """make 2 gausian distribution
    
    :param mean: 2*1 vector
    :param sigma: 2*2matrix
    :return: 2*1 vector
    """
    z = np.array([np.random.randn() for i in range(2)])
    u, d, v = np.linalg.svd(sigma)
    D = np.diag(np.sqrt(d))
    x = D @ z
    y = np.linalg.inv(u) @ x
    y += mean
    return y


def plot(mean, sigma):
    """

    :param mean: 2*1 vector
    :param sigma: 2*2matrix
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    x = np.array([multi_gausian(mean, sigma) for i in range(200)])
    y = np.array([np.random.multivariate_normal(mean, sigma) for i in range(200)])
    ax.scatter(x[:, 0], x[:, 1], c='red',label=u'自作関数')
    ax.scatter(y[:, 0], y[:, 1], c='blue',label='np.random.multivariate_normal')
    plt.legend()
    plt.show()
    # plt.savefig("multi_gausian.pgf")


if __name__ == '__main__':
    mean = [10, 20]
    sigma = np.array([[10, 1], [1, 20]])
    plot(mean, sigma)
