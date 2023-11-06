#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : population.py -- File to estimate the population development
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2020-12-23
# Time-stamp: <Mon 2021-03-08 10:40 juergen>
#
# Copyright (c) 2020 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from realoptions import GeometricBrownianMotion, OrnsteinUhlenbeckProcess


def load(filename: str, sep: str = ';') -> pd.DataFrame:
    """Load rainfall data as pandas data frame"""
    data = pd.read_csv(filename, sep=sep)
    data.columns = ['T', 'Y']
    return data


def main() -> None:
    """Run the main code for the population model"""
    data = load('./data/population.csv')

    #plt.plot(data['T'], data['Y'], label='population')
    # plt.show()
    # print(data['T'].iloc[0])

    data['T'] = data['T'] - data['T'].iloc[0]

    # print('xxxxxxxxx', data['Y'].iloc[0])
    X = GeometricBrownianMotion(0.007, 0.015, data['Y'].iloc[0])
    Y = OrnsteinUhlenbeckProcess(100000, 3000, .03, data['Y'].iloc[0])

    Y = OrnsteinUhlenbeckProcess(100000, 3000, .03, 64350)
    sim = Y.simulate(3)

    # plt.plot(sim)
    # plt.show()
    sim = Y.simulate(2000)

    ci = .95
    ci_u = (1-ci)/2+ci
    ci_l = (1-ci)/2

    _len = 61
    _start = - 30
    # print(y)
    # plt.plot(ci_u)
    # plt.plot(ci_l)
    fig, ax = plt.subplots(1, sharex=False, figsize=(8, 4))
    ax.fill_between(range(len(np.mean(sim, axis=1))),
                    np.quantile(sim, ci_u, axis=1),
                    np.quantile(sim, ci_l, axis=1),
                    color='k', alpha=.2, label='{}%-ci'.format(ci*100))
    ax.plot(np.mean(sim, axis=1), color='k', label='mean')
    ax.plot(sim[:, 0:1], linewidth=.7, label='example simulation')
    ax.plot([0, 14], [data['Y'].iloc[0],
                      data['Y'].iloc[0]+13000], color='orange', label='forecast')
    ax.plot(data['T'].head(10), data['Y'].head(10), label='recorded')
    # plt.plot(sim)
    ax.legend()
    ax.xaxis.set_ticks(np.arange(_start, _len, 5))
    ax.set_xticklabels([int(2021+x)
                        for x in np.arange(_start, _len, 5)], rotation=45)
    ax.set_xlim(_start, _len-0.5)

    ax.set_ylabel(r'[p]')
    ax.set_title('Population')
    plt.savefig('population.png', bbox_inches='tight')
    # print(data)

    # ax.set_xticklabels([int(2021+x)
    #                     for x in np.arange(0, _len, 5)], rotation=45)
    # fig = Y.plot(sim=10000, mean=True, show=False, ci=.95)
    # plt.show()


if __name__ == '__main__':
    main()


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
