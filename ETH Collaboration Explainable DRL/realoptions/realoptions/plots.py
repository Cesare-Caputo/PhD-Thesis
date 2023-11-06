"""Plot functions"""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plots.py -- Collection of plotting functions
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2021-02-12
# Time-stamp: <Thu 2021-04-22 09:55 juergen>
#
# Copyright (c) 2021 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
import csv
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
# plt.style.use('ggplot')


def plot(result, ax=None, ci=None, mean=True, show=False, bar=True, year=2021, xlabel=None, ylabel=None, title=None, levels=None, cum=False, cumci=False, loc=0, bar_color=None, ylim=None, ylim2=None, **kwargs):
    """Bar Plot"""

    data = result['data']
    labels = result['labels']
    _len = len(data)
    trajectories = np.sum(data, axis=1)

    if not ax:
        _, ax = plt.subplots()

    if cum:
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(np.mean(trajectories, axis=1)),
                 linewidth=2.0, label='cumulative')

        if cumci:
            cumci_u = (1-cumci)/2+cumci
            cumci_l = (1-cumci)/2
            ax2.fill_between(range(len(trajectories)),
                             np.cumsum(np.quantile(
                                 trajectories, cumci_u, axis=1)),
                             np.cumsum(np.quantile(
                                 trajectories, cumci_l, axis=1)),
                             color='k', alpha=.2, label='{}%-ci'.format(cumci*100))
        # ax2.set_yscale('log')
        # ax2.legend(loc=loc)
        if ylim2:
            ax2.set_ylim(ylim2[0], ylim2[1])
        else:
            ax2.set_ylim(0)
    if bar:
        values = np.mean(data, axis=2)
        df = pd.DataFrame(values, columns=labels)
        df.plot(ax=ax, kind="bar", stacked=True, color=bar_color)

    if show:
        if isinstance(show, int):
            ax.plot(trajectories[:, 0:show], label='example simulation')
        else:
            ax.plot(trajectories)
    if ci:
        ci_u = (1-ci)/2+ci
        ci_l = (1-ci)/2
        ax.fill_between(range(len(trajectories)),
                        np.quantile(trajectories, ci_u, axis=1),
                        np.quantile(trajectories, ci_l, axis=1),
                        color='k', alpha=.2, label='{}%-ci'.format(ci*100))

    if mean:
        ax.plot(np.mean(trajectories, axis=1), color='k', label='mean')

    if levels:
        for level, values in levels.items():
            y = values
            if isinstance(values, (int, float)):
                y = np.full(_len, values)
            ax.plot(y, label=level, linewidth=2.0,)

    ax.xaxis.set_ticks(np.arange(0, _len, 5))
    ax.set_xticklabels([int(year+x)
                        for x in np.arange(0, _len, 5)], rotation=45)
    ax.set_xlim(-.5, _len-0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    ax.legend(loc=loc)
    return ax


def plot_prob(result, ax=None, cdf=True, log=False, mean=False, ci=None, loc=0, xlabel=None, ylabel=None, title=None, label=None, ci_label=False, mean_label=False, xlim=None, color=None, **kwargs):
    data = result['data']
    labels = result['labels']
    _len = data.shape[2]

    values = np.sum(data, axis=(0, 1))

    if not ax:
        _, ax = plt.subplots()

    if cdf:
        X = np.sort(values)
        Y = np.array(range(_len))/float(_len)
        if label is None:
            label = 'cdf'
        ax.plot(X, Y, label=label, color=color)
        ax.set_ylim(0, 1)
    else:
        ax.hist(values, bins='auto', density=True)
        Y, x = np.histogram(values, bins='auto', density=True)
        X = x[:-1] + (x[1] - x[0])/2
        ax.plot(X, Y, label='pdf', color=color)

    if mean:
        m = np.mean(values)
        print('Option {}={}'.format(label, m))
        ax.axvline(x=m, ls='--', linewidth=1, color=color)
        if mean_label:
            ax.text(m, -.05, '{:.0f}'.format(m), color='red',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
    if ci:
        q_u = np.quantile(values, (1-ci)/2+ci)
        q_l = np.quantile(values, (1-ci)/2)
        ax.axvspan(q_l, q_u, alpha=0.2, color='k',
                   label='{}%-ci'.format(ci*100))
        if ci_label:
            ax.text(q_u, -.05, '{:.0f}'.format(q_u), color='orange',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
            ax.text(q_l, -.05, '{:.0f}'.format(q_l), color='orange',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
    if log:
        ax.set_xscale('log')

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if xlim:
        if isinstance(xlim, tuple):
            ax.set_xlim(xlim[0], xlim[1])
        elif isinstance(xlim, float):
            x_u = np.quantile(values, (1-xlim)/2+xlim)
            x_l = np.quantile(values, (1-xlim)/2)
            ax.set_xlim(x_l, x_u)

    ax.legend(loc=loc)
    return ax

def plot_prob_rl(result, ax=None, cdf=True, log=False, mean=False, ci=None, loc=0, xlabel=None, ylabel=None, title=None, label=None, ci_label=False, mean_label=False, xlim=None, color=None, **kwargs):
    data = result
    labels = ["DRL"]
    _len = len(data)

    values = np.sum(data, axis=(0, 1))

    if not ax:
        _, ax = plt.subplots()

    if cdf:
        X = np.sort(values)
        Y = np.array(range(_len))/float(_len)
        if label is None:
            label = 'cdf'
        ax.plot(X, Y, label=label, color=color)
        ax.set_ylim(0, 1)
    else:
        ax.hist(values, bins='auto', density=True)
        Y, x = np.histogram(values, bins='auto', density=True)
        X = x[:-1] + (x[1] - x[0])/2
        ax.plot(X, Y, label='pdf', color=color)

    if mean:
        m = np.mean(values)
        print('Option {}={}'.format(label, m))
        ax.axvline(x=m, ls='--', linewidth=1, color=color)
        if mean_label:
            ax.text(m, -.05, '{:.0f}'.format(m), color='red',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
    if ci:
        q_u = np.quantile(values, (1-ci)/2+ci)
        q_l = np.quantile(values, (1-ci)/2)
        ax.axvspan(q_l, q_u, alpha=0.2, color='k',
                   label='{}%-ci'.format(ci*100))
        if ci_label:
            ax.text(q_u, -.05, '{:.0f}'.format(q_u), color='orange',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
            ax.text(q_l, -.05, '{:.0f}'.format(q_l), color='orange',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top')
    if log:
        ax.set_xscale('log')

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if xlim:
        if isinstance(xlim, tuple):
            ax.set_xlim(xlim[0], xlim[1])
        elif isinstance(xlim, float):
            x_u = np.quantile(values, (1-xlim)/2+xlim)
            x_l = np.quantile(values, (1-xlim)/2)
            ax.set_xlim(x_l, x_u)

    ax.legend(loc=loc)
    return ax



def plot_d_and_s(result, name='', filename='supply_and_demand', fformat='pdf'):
    """Plot demand and supply"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, figsize=(16, 8))
    fig.suptitle('Supply and Demand for {}'.format(name))

    supply = np.sum(result['supply']['data'], axis=1, keepdims=True)
    demand = np.sum(result['demand']['data'], axis=1, keepdims=True)*-1
    colors = ['C{}'.format(i) for i in range(10)]

    delta = supply + demand
    # print(result['supply']['labels'])
    lables = result['supply']['labels']
    ashie = result['supply']['data'][:, [0], :]
    loch = result['supply']['data'][:, [-3], :]
    storage = result['supply']['data'][:, [-2], :]
    pumping = result['supply']['data'][:, [-1], :]

    dat = np.hstack([ashie, loch, pumping])
    lab = [lables[0], lables[-3], lables[-1]]

    # print(pumping)
    # print(demand.shape)
    _l = lables[1:-3]

    if len(_l) == 1:
        add = result['supply']['data'][:, [-4], :]
        add = demand*-1 - ashie-loch-pumping
        dat = np.hstack([dat, add])
        lab.append(_l[0])
        # print(add)

    sub = {'data': dat, 'labels': lab}
    # print(dat.shape)
    # print(lab)
    col1 = colors
    idx = result['supply']['data'].shape[1]
    col2 = colors[idx:] + colors[:idx]

    capacity = np.mean(np.sum(result['capacity']['data'], axis=1), axis=1)

    ax1 = plot(sub, ax=ax1, ci=.95,
               title='Supply', ylabel='Supply [Ml/d]',
               levels={'capacity': capacity}, bar_color=col1)

    # ax1 = plot(result['supply'], ax=ax1, ci=.95,
    #            title='Supply', ylabel='Supply [Ml/d]',
    #            levels={'capacity': capacity}, bar_color=col1)

    ax3 = plot(result['demand'], ax=ax3, ci=.95,
               title='Demand', ylabel='Demand [Ml/d]',
               levels={'capacity': capacity}, bar_color=col2)

    ax2 = plot(result['level'], ax=ax2, ci=.95,
               title='Water level', ylabel='[Ml]', bar=False,
               levels={'Top water level': 144402,
                       'Fish pass level': 144402-7373,
                       'Intake level': 144402-13646})

    ax4 = plot({'data': np.hstack([supply, demand]),
                'labels': ['supply', 'demand']}, ax=ax4, ci=.95,
               title='Supply - Demand', ylabel='[Ml/d]', bar=False,
               levels={'': 0})

    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.05,
                        right=.95, top=0.90, bottom=0.07)
    plt.savefig('{}.{}'.format(filename, fformat))
    plt.clf()


def plot_costs(result, name='', filename='costs', cumci=False, fformat='pdf'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(16/2, 8+4))
    fig.suptitle('Costs for {}'.format(name))

    data = result['cost']['data']
    labels = result['cost']['labels']
    colors = ['C{}'.format(i) for i in range(10)]

    ylim = (0, 40)
    ylim2 = (0, 1500)
    col1 = colors
    col2 = colors[3:] + colors[:3]

    ax1 = plot(result['cost'], ax=ax1, title='Total costs',
               ylabel='[m£]', cum=True, cumci=cumci, mean=False, bar_color=colors,
               ylim=ylim, ylim2=ylim2)

    ax2 = plot(result['compensation'], ax=ax2, title='Costs of inadequate service',
               ylabel='[m£]', cum=True, cumci=cumci, mean=False, bar_color=col1,
               ylim=ylim, ylim2=ylim2)

    ylim = (0, 30)
    ylim2 = (0, 150)

    ax3 = plot(result['investment'], ax=ax3, title='Intervention costs',
               ylabel='[m£]', cum=True, cumci=cumci, mean=False, bar_color=col2,
               ylim=ylim, ylim2=ylim2)

    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.1,
                        right=.93, top=0.90, bottom=0.07)

    plt.savefig('{}.{}'.format(filename, fformat))
    plt.clf()


def plot_risk(result, name='', filename='risk', fformat='pdf'):
    """Plot of the cdf and pdf of the risk."""
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16/2, 8))
    fig.suptitle('Costs and Risks for {}'.format(name))

    xlim = (0, 4000)
    ax1 = plot_prob(result['cost'], ax=ax1, title='CDF',
                    cum=True, mean=True,
                    ci=.98, mean_label=True,
                    ci_label=True,
                    xlim=xlim,
                    log=False,)

    ax2 = plot_prob(result['cost'], ax=ax2, title='PDF', cdf=False,
                    xlabel='[m£]', cum=True, mean=True, ci=.98)

    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.1,
                        right=.95, top=0.90, bottom=0.07)

    plt.savefig('{}.{}'.format(filename, fformat))
    plt.clf()


def plot_input(result, name='', filename='inputs', fformat='pdf'):
    """Plot demand and supply"""

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, sharex=False, figsize=(16/2, 8+8))

    # fig, ax4 = plt.subplots(1, sharex=False, figsize=(16/2, 4))
    fig.suptitle('Inputs for {}'.format(name))

    data = result['input']['data']
    labels = result['input']['labels']

    pop = {'data': data[:, [0], :], 'labels': labels[0]}
    pcc = {'data': data[:, [1], :], 'labels': labels[1]}
    pph = {'data': data[:, [2], :], 'labels': labels[2]}
    ind = {'data': data[:, [3], :], 'labels': labels[3]}
    eev = {'data': data[:, [4], :], 'labels': labels[4]}
    edm = {'data': data[:, [5], :], 'labels': labels[5]}
    pre = {'data': data[:, [6], :], 'labels': labels[6]}

    ax1 = plot(pop, ax=ax1, ci=.95, title='Population',
               ylabel='[p]', bar=False, show=3)

    ax2 = plot(pre, ax=ax2, ci=.95, title='Precipitation',
               ylabel=r'[$l/m^2/y$]', bar=False, show=3)

    ax3 = plot(ind, ax=ax3, ci=.95, title='Industry',
               ylabel=r'[Ml/d]', bar=False, show=1)

    ax4 = plot(eev, ax=ax4, ci=.95, title='External',
               ylabel=r'[Ml/d]', bar=False, show=1)

    # ax2 = plot(pcc, ax=ax2, ci=.95, title='Per capital consumption',
    #            ylabel='[l/h/d]', bar=False)
    # ax4 = plot(pph, ax=ax4, ci=.95, title='People per household',
    #            ylabel='[p/h]', bar=False)

    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.12,
                        right=.97, top=0.90, bottom=0.07)
    plt.savefig('{}.{}'.format(filename, fformat))
    #plt.savefig('external.png', bbox_inches='tight')
    plt.clf()


def plot_time(result, name='', filename='timing', fformat='pdf'):
    """Plot demand and supply"""

    data = result['timing']['data']
    labels = result['timing']['labels']
    colors = ['C{}'.format(i) for i in range(10)]
    p = len(labels)
    ylim = (0, 0.07)

    fig, axs = plt.subplots(p, sharex=True, figsize=(8, 4*p))
    fig.suptitle('Times for {}'.format(name))

    for i, ax in enumerate(axs):
        values = {'data': data[:, [i], :], 'labels': [labels[i]]}
        ax = plot(values, ax=ax, mean=False, bar_color=colors[i], ylim=ylim)

    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.12,
                        right=.97, top=0.90, bottom=0.07)
    plt.savefig('{}.{}'.format(filename, fformat))
    plt.clf()


def plot_risk_comp(results, name='', filename='risk_comp', fformat='pdf'):
    fig, ax = plt.subplots(1, sharex=False, figsize=(16/2, 8/2))
    # fig.suptitle(
    #     '{}'.format(name))
    colors = ['C{}'.format(i) for i in range(10)]

    colors.extend(['black', 'magenta', 'red'])
    print(colors)
    for i, (option, result) in enumerate(results.items()):
        ax = plot_prob(result['cost'], ax=ax, color=colors[i],
                       cum=True, mean=True, label=option, log=True)

    # ax2 = plot_prob(result['cost'], ax=ax2, title='PDF', cdf=False,
    #                 xlabel='[m£]', cum=True, mean=True, ci=.98)

    # fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.1,
    #                     right=.95, top=0.90, bottom=0.07)

    ax.set_xlabel(r'Costs [m£]')
    ax.set_ylabel('Cumulative probability')
    # ax.set_title(title)

    plt.savefig('{}.{}'.format(filename, fformat), bbox_inches='tight')
    plt.clf()




def cost_benefit(results, base=None, filename='summary', fformat='csv'):
    """csv with the cost benefit analysis"""
    _b = results.pop(base)

    base_risk = np.mean(np.sum(_b['cost']['data'][:, 0:3, :], axis=(0, 1)))
    base_cost = np.mean(np.sum(_b['cost']['data'][:, 3:, :], axis=(0, 1)))

    data = [['option', 'cost', 'risk', 'benefit', 'bcr', 'roi'],
            [base, base_cost, base_risk, None, None, None]]
    for key, opt in results.items():
        _risk = np.mean(np.sum(opt['cost']['data'][:, 0:3, :], axis=(0, 1)))
        _cost = np.mean(
            np.sum(opt['cost']['data'][:, 3:, :], axis=(0, 1))) - base_cost
        _benefit = base_risk - _risk
        bcr = _benefit/_cost
        roi = (_benefit-_cost)/_cost
        data.append([key, _cost, _risk, _benefit, bcr, roi])

    with open("{}.{}".format(filename, fformat), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def plot_nb_comp(results, name='', filename='net_comp', fformat='pdf'):
    fig, ax = plt.subplots(1, sharex=False, figsize=(16/2, 8/2))
    # fig.suptitle(
    #     '{}'.format(name))
    colors = ['C{}'.format(i) for i in range(10)]
    colors.extend(['black', 'magenta', 'red'])
    colors.pop(0)
    _b = results.pop('0')

    base_risk = np.sum(_b['cost']['data'][:, 0:3, :], axis=(0, 1))
    base_cost = np.mean(np.sum(_b['cost']['data'][:, 3:, :], axis=(0, 1)))

    _len = base_risk.shape[0]

    values = base_risk
    X_base = np.sort(values)
    n_values = 1000
    Y = np.array(range(n_values))/float(n_values)
    #ax.plot(X_base, Y)
    ax.set_ylim(0, 1)

    for i, (option, result) in enumerate(results.items()):
        color = colors[i]
        label = option
        risk = np.sum(result['cost']['data'][:, 0:3, :], axis=(0, 1))
        cost = np.mean(np.sum(result['cost']['data'][:, 3:, :], axis=(0, 1)))
        print(cost)
        risk = np.sort(risk)
        X = []
        for q in np.linspace(0, 1, n_values):
            X.append(np.quantile(X_base, q)-np.quantile(risk, q)-cost)
        X = np.sort(X)
        ax.plot(X, Y, label=label, color=color)

        ax.axvline(x=np.mean(X), ls='--', linewidth=1, color=color)

        print(np.mean(X))

    ax.legend(loc=None)
    ax.set_xlim(100, 3000)
    ax.set_xscale('log')
    ax.set_xlabel(r'Net benefit [m£]')
    ax.set_ylabel('Cumulative probability')
    # # ax.set_title(title)

    plt.savefig('{}.{}'.format(filename, fformat), bbox_inches='tight')
    plt.clf()


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
