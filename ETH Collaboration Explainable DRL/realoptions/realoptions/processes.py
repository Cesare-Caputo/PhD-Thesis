"""Stochastic processes"""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : processes.py -- Models for processes
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2020-12-17
# Time-stamp: <Tue 2021-02-16 11:39 juergen>
#
# Copyright (c) 2020 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
from typing import Optional, Any
from types import LambdaType
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


class BaseProcess:
    """Documentation for BaseProcess

    """

    def __init__(self, name: str = None, duration: float = 60.0,
                 start: float = 0.0, dt: float = 1.0, **kwargs):
        """Initialize the base process"""
        # name of the process
        self._name = name

        # duration of the process
        self._duration = duration

        # start of the process
        self._start = start

        # delta time step
        self._dt = dt

        self._time = np.arange(start, duration+dt, step=dt)

        self._kwargs = kwargs

    @property
    def name(self) -> Optional[str]:
        """Return the name of the process."""
        return self._name

    @property
    def duration(self) -> float:
        """Return the duration of the process."""
        return self._duration

    @property
    def time(self) -> float:
        """Return the time vector of the process."""
        return self._time

    def _get_time(self, duration: float = None, start: float = None,
                  delta: float = None) -> Any:
        """Helper function to get the time vector"""

        if not start and not duration and not delta:
            time = self._time
        else:
            # check variables
            if not start:
                start = self._start
            if not duration:
                duration = self._duration
            if not delta:
                delta = self._dt
            time = np.arange(start, duration+delta, step=delta)

        return time

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""
        raise NotImplementedError

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        """Plot a simulation run"""
        return plt.plot(self.time, self.simulate(), *args,
                        scalex=scalex, scaley=scaley, data=data, **kwargs)


class DeterministicProcess(BaseProcess):
    """Class of a deterministic process"""

    def __init__(self, function, **kwargs: Any):
        """Initialize the base process"""

        super().__init__(**kwargs)

        self._function = function

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)

        # get time
        time = self._get_time(duration, start, delta)

        if isinstance(self._function, (int, float)):
            result = np.full((len(time), simulations), self._function)

        elif isinstance(self._function, LambdaType):
            result = self._function(time)

        else:
            result = np.zeros((len(time), simulations))

        return result


class GeometricBrownianMotion(BaseProcess):
    """Class of a geometric brownian motion"""

    def __init__(self, mu: float, sigma: float, value: float = 1., **kwargs):
        """Initialize the base process"""

        super().__init__(**kwargs)

        self._mu = mu
        self._sigma = sigma
        self._value = value

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)
        if not delta:
            delta = self._dt

        # get time
        time = self._get_time(duration, start, delta)

        # random variables
        normal = np.random.standard_normal((len(time), simulations))

        # standard brownian motion
        brownian_motion = np.cumsum(normal, axis=0)*np.sqrt(delta)

        # drift term
        drift = np.tile((self._mu-0.5*self._sigma**2)*time, (simulations, 1)).T

        # volatility
        volatility = self._sigma * brownian_motion

        result = self._value * (np.exp(drift + volatility))

        return result

    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        """Plot a simulation run"""
        # get kwargs
        sim = kwargs.pop('sim', 1)
        mean = kwargs.pop('mean', False)
        show = kwargs.pop('show', False)
        ci = kwargs.pop('ci', False)

        # simulate results
        result = self.simulate(sim)

        _, ax = plt.subplots()

        if ci:
            ci_u = (1-ci)/2+ci
            ci_l = (1-ci)/2
            ax.fill_between(self.time,
                            np.quantile(result, ci_u, axis=1),
                            np.quantile(result, ci_l, axis=1),
                            color='k', alpha=.2, label='{}%-ci'.format(ci*100))

        if show:
            ax.plot(self.time, result, *args,
                    scalex=scalex, scaley=scaley, data=data, **kwargs)

        if mean:
            ax.plot(self.time, np.mean(result, axis=1),
                    '-k', label='mean', linewidth=2.0)

            ax.legend(loc=2)
        return ax


class OrnsteinUhlenbeckProcess(GeometricBrownianMotion):
    """Class of a geometric brownian motion"""

    def __init__(self, mu: float, sigma: float, theta: float,
                 value: float = 0.0, **kwargs):
        """Initialize the base process"""

        super().__init__(mu=mu, sigma=sigma, value=value, **kwargs)

        self._theta = theta

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)
        if not delta:
            delta = self._dt

        # get time
        time = self._get_time(duration, start, delta)

        # initialize output vector
        y = np.zeros((len(time), simulations))

        # initial condition
        y[0] += self._value

        # define drift term
        def drift(value):
            return self._theta*(self._mu-value)

        # def diffusion(y, t): return self._sigma  # define diffusion term

        # define noise process
        noise = np.random.normal(
            loc=0.0, scale=1.0, size=(len(time), simulations))*np.sqrt(delta)

        # solve SDE
        for i in range(1, len(time)):
            y[i] = y[i-1] + drift(y[i-1]) * delta + self._sigma * noise[i]

        return y


class LognormalProcess(GeometricBrownianMotion):
    """Independent lognormal process"""

    def __init__(self, mu: float, sigma: float, value: float = 1.,
                 mu_shift: float = 0, sigma_shift: float = 0, **kwargs):
        """Initialize the base process"""

        super().__init__(mu, sigma, **kwargs)
        self._mu_shift = mu_shift
        self._sigma_shift = sigma_shift

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)
        if not delta:
            delta = self._dt

        # get time
        time = self._get_time(duration, start, delta)

        # random variables
        scale_m = 1+np.cumsum(np.full(len(time), self._mu_shift))
        scale_s = 1+np.cumsum(np.full(len(time), self._sigma_shift))

        scale = np.log(np.full(len(time), self._mu) * scale_m)
        sigma = np.full(len(time), self._sigma) * scale_s

        stack = np.vstack([scale, sigma]).T

        result = np.vstack([np.random.lognormal(
            x[0], x[1], size=simulations) for x in stack])

        return result


class NormalProcess(GeometricBrownianMotion):
    """Independent normal process"""

    def __init__(self, mu: float, sigma: float, value: float = 1.,
                 mu_shift: float = 0, sigma_shift: float = 0, **kwargs):
        """Initialize the base process"""

        super().__init__(mu, sigma, **kwargs)
        self._mu_shift = mu_shift
        self._sigma_shift = sigma_shift

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)
        if not delta:
            delta = self._dt

        # get time
        time = self._get_time(duration, start, delta)

        # random variables
        scale_m = 1+np.cumsum(np.full(len(time), self._mu_shift))
        scale_s = 1+np.cumsum(np.full(len(time), self._sigma_shift))

        mu = np.full(len(time), self._mu) * scale_m
        sigma = np.full(len(time), self._sigma) * scale_s

        stack = np.vstack([mu, sigma]).T

        result = np.vstack([np.random.normal(
            x[0], x[1], size=simulations) for x in stack])

        return result


class PoissonPointProcess(BaseProcess):
    """Class of a deterministic process"""

    def __init__(self, lam, **kwargs: Any):
        """Initialize the base process"""

        super().__init__(**kwargs)

        self._lambda = lam

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get arguments
        simulations: int = 1
        if args:
            simulations = args[0]

        # get variables
        duration = kwargs.get('duration', None)
        start = kwargs.get('start', None)
        delta = kwargs.get('dt', None)

        # get time
        time = self._get_time(duration, start, delta)

        result = np.random.poisson(self._lambda, (len(time), simulations))

        return result


class MarkedPoissonPointProcess(PoissonPointProcess):
    """Class of a deterministic process"""

    def __init__(self, lam, mark=2., parameters=None, **kwargs: Any):
        """Initialize the base process"""

        super().__init__(lam, **kwargs)

        self._mark = mark
        self._parameters = parameters

    def simulate(self, *args, **kwargs) -> Any:
        """Simulate the process"""

        # get poisson point process
        ppp = super().simulate(*args, **kwargs)

        result = np.zeros(ppp.shape)

        # apply mark to the process
        for i in range(1, np.max(ppp)+1):
            _len = len(ppp[ppp == i])
            if isinstance(self._mark, (int, float)):
                values = sum([self._mark]*i)
            else:
                values = np.sum(self._mark(
                    **self._parameters, size=(_len, i)), axis=1)

            result[ppp == i] = values

        return result


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
