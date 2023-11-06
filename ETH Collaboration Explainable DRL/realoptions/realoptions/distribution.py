"""Modul to get the distribution over time"""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : distribution.py -- Modul to get the distribution over time
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2020-12-09
# Time-stamp: <Mon 2020-12-14 10:01 juergen>
#
# Copyright (c) 2020 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
import random  # type: ignore
import numpy as np  # type: ignore
from scipy.stats import norm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


class Distribution:
    """Documentation for Distribution

    """

    def __init__(self, lifetime: int = 70,
                 mu: float = 0.0166065364639344,
                 mu_years: int = 30,
                 mu_increments: float = 0.0001,
                 mu_expert=None,
                 # mu_expert: dict = {1: 0.002,
                 #                    2: 0.0025,
                 #                    3: 0.003,
                 #                    4: 0.004,
                 #                    5: 0.007,
                 #                    6: 0.01,
                 #                    7: 0.013,
                 #                    8: 0.0138,
                 #                    9: 0.0142,
                 #                    10: 0.014606536463935},
                 sigma: float = 0.001,
                 sigma_years: int = 30,
                 sigma_increments: float = 0.0,
                 sigma_expert=None,
                 # sigma_expert: dict = {1: 0.0004,
                 #                       2: 0.0004,
                 #                       3: 0.0004,
                 #                       4: 0.0004,
                 #                       5: 0.0004,
                 #                       6: 0.0004,
                 #                       7: 0.0007,
                 #                       8: 0.0009,
                 #                       9: 0.0010,
                 #                       10: 0.0010},
                 ) -> None:
        """Initialize the class"""
        # variables
        # lifetime of the random variable
        self.lifetime = lifetime

        # delta time in years
        self.delta_t = 1

        # time of the random variable (up to the lifetime)
        self.time = np.arange(1, self.lifetime+self.delta_t, self.delta_t)

        # delta probabilities
        self.delta_y = 0.0005

        # probabilities
        self.y = np.arange(0, 1+self.delta_y, self.delta_y)

        # mu vector
        self.mu_vec = mu + (self.time-mu_years)*mu_increments

        if mu_expert:
            for key, value in mu_expert.items():
                self.mu_vec[np.where(self.time == key)] = value

        # sigma vector
        self.sigma_vec = sigma + (self.time-sigma_years)*sigma_increments

        if sigma_expert:
            for key, value in sigma_expert.items():
                self.sigma_vec[np.where(self.time == key)] = value

        # make distribution matrix
        self.matrix = np.ones((len(self.time), len(self.y)))

        for i in range(len(self.time)):
            self.matrix[i] = norm.cdf(self.y, self.mu_vec[i], self.sigma_vec[i])

        # print(len(norm.cdf(self.y, self.mu_vec[0], self.sigma_vec[0])))
        # print(self.matrix)
        plt.imshow(np.transpose(self.matrix), extent=[
                   1, 70, 0, 1], aspect=70, origin='lower')
        plt.colorbar()
        plt.show()

        # print(len(self.y))
        # value = 0.027930845492184
        # value = 0.916069866771696
        # pdf = np.zeros(len(self.time))
        # idx = (np.abs(self.matrix[0] - value)).argmin()
        # # print(idx)
        # # print(self.y[idx])

        # for i in range(len(pdf)):
        #     idx = (np.abs(self.matrix[i] - value)).argmin()
        #     if self.matrix[i][idx] > value:
        #         idx = idx-1
        #     pdf[i] = self.y[idx]
        # cdf = np.zeros(len(self.time))
        # cdf[0] = pdf[0]
        # cdf[1:] = np.cumsum(pdf[1:], axis=0)
        # # print(len(pdf))
        # # print(cdf)
        # # print(len(cdf))

        # threshold = 83.5030425921881 / 100
        # threshold = 23.1773516836107 / 100
        # idx = (np.abs(cdf - threshold)).argmin()
        # if cdf[idx] < threshold:
        #     idx = idx+1
        # print(self.time[idx])

        # # Simulation

        # num_of_sim = 1000
        # sim = np.zeros((num_of_sim, 2))

        # for j in range(num_of_sim):
        #     value = random.random()
        #     pdf = np.zeros(len(self.time))
        #     cdf = np.zeros(len(self.time))
        #     idx = (np.abs(self.matrix[0] - value)).argmin()

        #     for i in range(len(pdf)):
        #         idx = (np.abs(self.matrix[i] - value)).argmin()
        #         if self.matrix[i][idx] > value:
        #             idx = idx-1
        #         pdf[i] = self.y[idx]

        #     cdf[0] = pdf[0]
        #     cdf[1:] = np.cumsum(pdf[1:], axis=0)

        #     threshold = random.random()

        #     # plt.plot(cdf)
        #     # plt.show()

        #     idx = (np.abs(cdf - threshold)).argmin()
        #     if cdf[idx] < threshold:
        #         idx = idx+1
        #     try:
        #         sim[j][0] = self.time[idx]
        #         sim[j][1] = 1.
        #     except IndexError:
        #         sim[j][0] = 999
        #         sim[j][1] = 0.

        # print('probability of change', np.average(sim, axis=0)[1])
        # print('average year of change', np.average(sim[sim[:, 0] < 999, 0]))
        # print('std of change', np.std(sim[sim[:, 0] < 999, 0]))
        # plt.hist(sim[sim[:, 0] < 999, 0])
        # plt.show()

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
