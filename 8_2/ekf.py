#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for problem 8-2
Extended Kalman Filter
"""
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'IPAPGothic'
import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import norm
from scipy.optimize import minimize


class EKF():
    """Extended Kalman Filter"""
    observer_1 = np.array([0, 100])
    observer_2 = np.array([100, 0])
    observer_3 = np.array([0, 0])
    u = 2

    def __init__(self, q, r):
        """Constructor

        :param q: \omega_t \sim N(0,Q)
        :param r: v_t \sim N(0,R)
        """
        self.q = q
        self.r = r
        # estimation
        self.mu = np.array([0, 0])  # initial estimation
        self.variance = np.array([[0, 0], [0, 0]])  # initial variance

    def jacobian(self, mu):
        """ calculate jacobian

        :param mu: estimation of coord
        :return:jacobian
        """
        x = mu
        jacobian = [(x - EKF.observer_1) / norm(x - EKF.observer_1),
                    (x - EKF.observer_2) / norm(x - EKF.observer_2),
                    (x - EKF.observer_3) / norm(x - EKF.observer_3)]
        return np.array(jacobian)

    @classmethod
    def getdistance(cls, coord):
        """get distance from 3 points

        :param coord: coordinate of robot
        :return:
        """
        y1 = norm(coord - EKF.observer_1)
        y2 = norm(coord - EKF.observer_2)
        y3 = norm(coord - EKF.observer_3)
        y = np.array([y1, y2, y3]).T
        return y

    def prediction(self):
        """ prediction step

        :return: new_mu
        :return: new_variance
        """
        A = np.array([[1, 0], [0, 1]])
        B = 1
        new_mu = A @ self.mu + B * EKF.u
        new_variance = A @ self.variance @ A.T + self.q
        self.mu = new_mu
        self.variance = new_variance
        return new_mu, new_variance

    def update(self, y):
        """ observe and update step

        :param y: observation data
        :return:
        """
        C = self.jacobian(self.mu)
        V = self.variance
        R = self.r
        K = V @ (C.T) @ np.linalg.inv(C @ V @ (C.T) + R)  # Kalman Gain
        new_mu = self.mu + K @ (y - EKF.getdistance(self.mu))
        new_variance = (1 - K @ C) @ self.variance
        self.mu = new_mu
        self.variance = new_variance
        return new_mu, new_variance

    @classmethod
    def simple_estimation(cls, y):
        """ simple estimation not using kalman filter

        :param y:observation data
        :return:estimated coordinate
        """
        func = lambda coord: norm(EKF.getdistance(coord) - y)
        x0 = np.array([0, 0])
        res = minimize(func, x0, method='Powell')
        return res.x


def makefig(x, esti, simp):
    """make figure

    :param x: np array
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("x")
    ax.set_xlim(0, 220)
    ax.set_ylabel("y")
    ax.set_ylim(0, 220)
    ax.scatter(x[:, 0], x[:, 1], label="true course")
    ax.scatter(esti[:, 0], esti[:, 1], label="EKF")
    ax.scatter(simp[:, 0], simp[:, 1], label="simple estimation")
    plt.legend()
    plt.show()


def main():
    q = np.array([[3, 0], [0, 3]])
    r = np.array([[20, 10, 10], [10, 20, 10], [10, 10, 20]])
    filter = EKF(q, r)
    T = 100  # sec
    coord = np.array([0, 0])
    coords = [coord]
    estimation = [coord]
    simple_estimation = [coord]
    for i in range(T):
        coord = coord + EKF.u + multivariate_normal([0, 0], filter.q)
        coords.append(coord)
        y = EKF.getdistance(coord) + multivariate_normal([0, 0, 0], r)
        # Extended Kalman Filter
        filter.prediction()
        mu, var = filter.update(y)
        estimation.append(mu)
        # Simple Estimation
        simple_est = EKF.simple_estimation(y)
        simple_estimation.append(simple_est)

    makefig(np.array(coords), np.array(estimation), np.array(simple_estimation))


if __name__ == '__main__':
    main()
