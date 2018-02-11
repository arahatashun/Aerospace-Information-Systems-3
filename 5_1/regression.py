#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for problem 5-1
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
from math import log


def aic(q, n, k):
    """Akaike information criterion

    :param q:the sum of the squared error
    :param n:number of data
    :param k:number of parameter
    :return:AIC
    """
    aic = n * log(q / n) + 2 * k
    return aic


def read_data():
    """read data from mazdas.txt .

    :return year: list of year
    :return prices: list of price
    """
    years = []
    prices = []
    with open('mazdas.txt', 'r') as file:
        i = 0
        for line in file:
            items = line.strip().split('\t')
            if i == 0:
                # skip column　name
                i = i + 1
                pass
            else:
                years.append(int(items[0]))
                prices.append(int(items[1]))
    return years, prices


def plot_figure(years, prices):
    """ make matplotlib figure for overview

    :param years:
    :param prices:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(years, prices)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    plt.savefig("overview.pgf")


def plot_poly_fit(years, prices, order):
    """ linear regression

    :param years:
    :param prices:
    :param order: Degree of the fitting polynomial
    :return:
    """
    coefficient = np.polyfit(years, prices, order)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(years, prices)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    expression_string = "y="
    for i in range(order + 1):
        expression_string += "+" + str(coefficient[i]) + "$x^" + str(i) + "$\n"
    x_min = min(years)
    x_max = max(years)
    x = np.linspace(x_min, x_max, 100)
    ax.plot(x, np.polyval(coefficient, x), label=expression_string)
    ax.set_title("order" + str(order) + " regression")
    ax.legend(loc="upper left")
    sum_squared_error = sum([(prices[i] - np.polyval(coefficient, years[i])) ** 2 for i in range(len(years))])
    AIC = aic(sum_squared_error, len(years), order)
    ax.text(72, 10000, "AIC:" + str(AIC))
    # plt.show()
    # plt.savefig("aic" + str(order)+"_poly.pgf")
    return sum_squared_error


def calc_loo(years, prices, n):
    """ calc average squared error using leave-one-out method

    :param n: order
    :return:
    """
    squared_error = 0
    for i in range(len(years)):
        tmp_years = copy.deepcopy(years)
        tmp_prices = copy.deepcopy(prices)
        tmp_years.pop(i)
        tmp_prices.pop(i)
        coefficient = np.polyfit(tmp_years, tmp_prices, n)
        squared_error += (prices[i] - np.polyval(coefficient, years[i])) ** 2 / len(prices)
    print(squared_error)
    return squared_error


def print_loo_list(years, prices, highest_order):
    """ LOO cross-validation 0 to highest_order

    :param years:
    :param prices:
    :param highest_order:
    :return:
    """
    for i in range(highest_order + 1):
        print(i, calc_loo(years, prices, i))


if __name__ == '__main__':
    years, prices = read_data()
    n = input("order")
    print_loo_list(years, prices, n)
    # plot_figure(years, prices)
