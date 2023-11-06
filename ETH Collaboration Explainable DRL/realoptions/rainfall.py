#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : rainfall.py -- Inverness Rainfall Model
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2020-12-14
# Time-stamp: <Mon 2021-03-08 10:40 juergen>
#
# Copyright (c) 2020 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
import warnings
import itertools
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pandas.plotting import autocorrelation_plot  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.api import SARIMAX  # type: ignore
import numpy as np  # type: ignore
from scipy import stats
from realoptions import LognormalProcess, OrnsteinUhlenbeckProcess, NormalProcess


def load(filename: str, sep: str = ';') -> pd.DataFrame:
    """Load rainfall data as pandas data frame"""
    data = pd.read_csv(filename, sep=sep)
    data.columns = ['T', 'Y']
    return data


def plot_moving_average(data, period: int = 5, filename=None) -> None:
    """Plot the moving average of the data"""

    # calculate moving averages
    data['SMA'] = data.iloc[:, 1].rolling(window=period).mean()
    data['CMA'] = data.iloc[:, 1].expanding(min_periods=period).mean()
    data['EMA'] = data.iloc[:, 1].ewm(span=40, adjust=False).mean()

    # create plot
    plt.figure(figsize=[15, 10])
    plt.grid(True)
    plt.plot(data['T'], data['Y'], label='rainfall')
    plt.plot(data['T'], data['SMA'],
             label='Simple Moving Average {} Years'.format(period))
    plt.plot(data['T'], data['CMA'],
             label='Cumulative Moving Average {} Years'.format(period))
    plt.plot(data['T'], data['EMA'], label='Exponential Moving Average')
    plt.legend(loc=2)

    # save or show plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.clf()


def arima(data) -> None:
    """Fit an ARIMA model and plot residual errors"""
    series = data['Y']
    # fit model
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())


def arima_predict(data) -> None:
    """Fit an ARIMA model and plot residual errors"""
    series = data['Y']

    # split into train and test sets
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    # rmse = np.sqrt(mean_squared_error(test, predictions))
    # print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


def gbm(data):
    """Geometric Brownian Motion
    Parameter Definitions

    So    :   initial stock price
    dt    :   time increment -> a day in our case
    T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
    N     :   number of time points in prediction the time horizon -> T/dt
    t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
    mu    :   mean of historical daily returns
    sigma :   standard deviation of historical daily returns
    b     :   array for brownian increments
    W     :   array for brownian path
    """
    series = data['Y']
    X = series.values
    V = (X-np.mean(X))/np.mean(X)/10
    # print(V)
    mu = np.mean(V)
    print(mu)

    sigma = np.std(V)
    print(sigma)

    So = X[0]
    T = len(X)
    dt = 1
    N = T / dt
    t = np.arange(1, int(N) + 1)

    scen_size = 1000  # User input
    b = {str(scen): np.random.normal(0, 1, int(N))
         for scen in range(1, scen_size + 1)}
    W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

    # Calculating drift and diffusion components
    drift = (mu - 0.5 * sigma**2) * t
    # print(drift)
    diffusion = {str(scen): sigma * W[str(scen)]
                 for scen in range(1, scen_size + 1)}
    # print(diffusion)

    # Making the predictions
    S = np.array([So * np.exp(drift + diffusion[str(scen)])
                  for scen in range(1, scen_size + 1)])
    # print(S)
    # add So to the beginning series
    S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))

    # Plotting the simulations
    plt.figure(figsize=(20, 10))
    for i in range(scen_size):
        plt.title("Yearly Volatility: " + str(sigma))
        plt.plot(S[i, :])
        plt.ylabel('Annual Rainfall [mm]')
        plt.xlabel('Prediction Years')
    plt.plot(X, label='rainfall')
    plt.legend(loc=2)
    plt.show()


def sarimax(data):

    series = data['Y']
    # Y = data.set_index('T')

    # split into train and test sets
    Y = series  # .values

    model = ARIMA(Y,
                  order=(12, 1, 1),
                  # seasonal_order=(0, 2, 2, 12),
                  # enforce_stationarity=False,
                  # enforce_invertibility=False,
                  )
    results = model.fit()

    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    pred = results.get_prediction(start=80, dynamic=False)
    pred_ci = pred.conf_int()

    ax = Y.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Years')
    ax.set_ylabel('Annual Rainfall [mm]')
    plt.legend()

    # plt.show()
    # plt.clf()

    # plt.plot(data['Y'], label='observed')
    # # plt.plot(predictions, color='red')
    # # plt.show()

    # print(len(list(range(60, len(Y)))))

    # plt.plot(range(60, len(Y)), pred.predicted_mean)
    # #ax = data['Y'].plot(label='observed')
    # #pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
    # # print(len(pred.predicted_mean))
    # # ax.set_xlabel('Date')
    # # ax.set_ylabel('CO2 Levels')
    # plt.legend()

    # plt.show()

    # print(pred_ci)

    # Get forecast 500 steps ahead in future
    pred_uc = results.get_forecast(steps=70)

    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()

    # print(pred_ci)

    ax = Y.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    results.simulate(70, anchor='end').plot(ax=ax, label='Simulation 1')
    results.simulate(70, anchor='end').plot(ax=ax, label='Simulation 2')
    results.simulate(70, anchor='end').plot(ax=ax, label='Simulation 3')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annual Rainfall [mm]')

    plt.legend()
    plt.show()

    # print(sim)


def sarimax_parameters(data):

    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    series = data['Y']
    # split into train and test sets
    Y = series  # .values

    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12)
                    for x in list(itertools.product(p, d, q))]

    out = {}
    # 1,1,1,0,2,2,12
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = SARIMAX(Y,
    #                           order=param,
    #                           seasonal_order=param_seasonal,
    #                           enforce_stationarity=False,
    #                           enforce_invertibility=False)

    #             results = mod.fit()

    #             out['ARIMA{}x{}12'.format(param, param_seasonal)] = results.aic
    #         except:
    #             continue

    p_values = [9, 10, 11, 12, 13]
    d_values = range(0, 5)
    q_values = range(0, 5)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                print(order)
                try:
                    mod = ARIMA(Y,
                                order=order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

                    results = mod.fit()

                    out['ARIMA({}, {}, {})'.format(p, d, q)] = results.aic
                except:
                    continue

    for k, v in out.items():
        print(k, v)

    print(min(out, key=out.get))


def main() -> None:
    """Run the main code for the rainfall model"""
    data = load('./data/rainfall.csv')

    # plot_moving_average(
    #     data, period=10, filename='rainfall_moving_averages.pdf')
    # print(data.head())

    # autocorrelation_plot(data['Y'])
    # plt.show()

    # arima(data)
    # arima_predict(data)

    # gbm(data)
    # sarimax(data)

    # # plt.show()
    # s, loc, scale = stats.lognorm.fit(data['Y'], floc=0)
    # print(s, loc, scale)

    # x = stats.lognorm.rvs(s=s*1.5, loc=loc, scale=scale, size=10000)
    # #plt.hist(x, density=True)
    # # data['Y'].hist(density=True)
    # X = LognormalProcess(1430, 0.11516, mu_shift=-.001, sigma_shift=.005)
    # sim = X.simulate(10000)
    # print(np.mean(sim, axis=1))
    # # print(sim)
    # # plt.hist(sim, density=True)
    # # plt.show()
    # # plt.clf()
    # # sarimax_parameters(data)

    # # print(1000*5546219/10**6/365)

    # # Y = OrnsteinUhlenbeckProcess(100000, 3000, .03, data['Y'].iloc[0])
    # # sim = Y.simulate(3)
    # # print(sim)

    # fig = X.plot(sim=20, mean=True, show=True, ci=.95)
    # plt.show()

    data['T'] = data['T'] - 2020  # data['T'].iloc[0]

    # print('xxxxxxxxx', data['Y'].iloc[0])
    Y = LognormalProcess(1430, 0.11516, mu_shift=-.001, sigma_shift=.005)
    # Y = NormalProcess(1430, 140, mu_shift=-.001, sigma_shift=.005)
    sim = Y.simulate(2000)

    # y = np.mean(sim, axis=1)
    # ci_u = np.quantile(sim, 0.975, axis=1)
    # ci_l = np.quantile(sim, 0.275, axis=1)

    ci = .95
    ci_u = (1-ci)/2+ci
    ci_l = (1-ci)/2

    _len = 61
    _start = - 110
    _step = 10

    fig, ax = plt.subplots(1, sharex=False, figsize=(8, 4))
    ax.fill_between(range(len(np.mean(sim, axis=1))),
                    np.quantile(sim, ci_u, axis=1),
                    np.quantile(sim, ci_l, axis=1),
                    color='k', alpha=.2, label='{}%-ci'.format(ci*100))
    ax.plot(np.mean(sim, axis=1), color='k', label='mean')
    ax.plot(sim[:, 0:1], linewidth=.7, label='example simulation')

    ax.plot(data['T'], data['Y'], label='recorded')
    #plt.plot([0, 14], [data['Y'].iloc[0], data['Y'].iloc[0]+13000])
    # plt.plot(y)
    # plt.plot(ci_u)
    # plt.plot(ci_l)
    # plt.plot(Y.simulate(1))

    ax.legend()
    ax.xaxis.set_ticks(np.arange(_start, _len, _step))
    ax.set_xticklabels([int(2021+x)
                        for x in np.arange(_start, _len, _step)], rotation=45)
    #ax.set_xlim(_start, _len-0.5)

    ax.set_ylabel(r'[$l/m^2/y$]')
    ax.set_title('Precipitation')

    plt.savefig('rainfall.png', bbox_inches='tight')
    # print(data)


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
