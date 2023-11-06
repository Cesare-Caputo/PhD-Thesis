"""Options for the evaluation"""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : options.py -- methods to store options
# Author    : Jürgen Hackl <hackl@ifi.uzh.ch>
# Time-stamp: <Thu 2021-02-18 12:25 juergen>
#
# Copyright (c) 2016-2020 Pathpy Developers
# =============================================================================
from typing import Optional, Union, Any, cast, Iterable
import numpy as np


class BaseOption:
    """Class to make options"""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, name: Optional[str] = None,
                 dynamic: bool = False,
                 discount_rate: float = 0.0,
                 start_year: float = 0.0,
                 end_year: float = 61.0,
                 dt: float = 1.0,
                 **kwargs: Any) -> None:
        """Initialize base option"""
        # pylint: disable=too-many-arguments

        # Variables
        # name of the option
        self._name = name

        # discount rate
        self._discount_rate = discount_rate

        # if the option can be used dynamically
        # i.e. the start time may vary due to optimization
        self._dynamic = dynamic

        # start of the option
        self._start = kwargs.get('start', 0)

        # store the parameters
        self._parameters = kwargs

        # Initialize matrices
        # time vector
        self._time = np.arange(start_year, end_year, step=dt)

        # TODO: make the impact and costs more flexible
        # e.g. add additional costs or more detailed costs
        # as well as multiple impacts per option
        # impact of the option
        self._values = np.zeros((2, len(self)))

        # cost of the option
        # 0 - enhancement
        # 1 - replacement
        # 2 - opex
        self._costs = np.zeros((3, len(self)))

        # Update the values if kwargs are given
        if kwargs:
            self.set_values(**self._parameters)
            self.set_costs(**self._parameters)

    def __len__(self) -> int:
        """Return the lengt of the option in terms of columns"""
        return len(self._time)

    @property
    def name(self) -> Optional[str]:
        """Return the name of the option."""
        return self._name

    @property
    def start(self) -> int:
        """Return the start time of the option."""
        return self._start

    @property
    def dynamic(self) -> bool:
        """Return if the option can be used dynamically."""
        return self._dynamic

    @property
    def discount(self):
        """Returns a discount vector"""
        # TODO: account also for smaller and bigger steps than 1 year
        return np.ones((1, len(self._time))) /\
            (1+self._discount_rate) ** self._time

    def reset(self):
        """Reset the variables of the option"""
        self._values = np.zeros((2, len(self)))
        self._costs = np.zeros((3, len(self)))

    def set_values(self, increase: float = 0.0,
                   reduction: float = 0.0,
                   start: int = 0,
                   end: int = None,
                   duration: Optional[Union[int, tuple]] = None,
                   **kwargs) -> None:
        """Set values of the Option"""
        # pylint: disable=unused-argument
        # pylint: disable=too-many-arguments

        # check the duration of the enhancement
        if isinstance(duration, tuple) and isinstance(duration[0], int):
            _duration: int = duration[0]
        elif isinstance(duration, int):
            _duration = duration
        else:
            _duration = 0

        # assign values
        self._values[0][start+_duration:end] = increase
        self._values[1][start+_duration:end] = reduction

    def set_costs(self, enhancement: float = 0.0,
                  replacement: float = 0.0,
                  opex: float = 0.0,
                  start: int = 0,
                  end: int = None,
                  duration: Optional[Union[int, tuple]] = None,
                  interval: Optional[Union[int, tuple]] = None,
                  ** kwargs) -> None:
        """Set costs of the Option"""
        # pylint: disable=unused-argument
        # pylint: disable=too-many-arguments

        # create temporal variable to store the costs
        costs = [enhancement, replacement, opex]

        # check the duration
        if isinstance(duration, (int, type(None))):
            duration = (duration, None, None)

        # convert to list
        _duration = list(cast(Iterable[Any], duration))

        # check the interval
        if isinstance(interval, (int, type(None))):
            interval = (interval, None, None)

        # convert to list
        _interval = list(cast(Iterable[Any], interval))

        # iterate over the intervals
        for j, i in enumerate(_interval):

            # if duration is given split the costs over time
            if _duration[j] and _duration[j] > 0:
                costs[j] /= _duration[j]

            # if intervals are considered
            if i:
                # check the duration or use 0
                if not _duration[j]:
                    _duration[j] = 1

                # assign costs in intervals
                for k in np.arange(start, len(self), i):
                    self._costs[j][k:k + _duration[j]] += costs[j]
            # if no intervals are considered
            else:
                # if duration is given assign costs once
                if _duration[j]:
                    self._costs[j][start:start + _duration[j]] += costs[j]

                # if no duration is given apply costs till the end
                # thereby the costs are assigned after the initial duration
                else:
                    if not _duration[0]:
                        _duration[0] = 0
                    self._costs[j][start + _duration[0]:end] += costs[j]

    def costs(self, discounted: bool = False, total: bool = True):
        """Function to return the cost matrix of the option"""
        costs = self._costs.copy()
        if discounted:
            costs *= self.discount
        if total:
            costs = np.sum(costs, axis=0).reshape(len(self._time), 1).T

        return costs.T

    def npv(self, total: bool = True):
        """Return the net present value of the option"""
        costs = self.costs(discounted=True, total=total)
        return np.sum(costs)

    def reduction(self):
        """Reduction of the demand"""
        return self._values[[1]].T

    def increase(self):
        """Increase of the supply"""
        return self._values[[0]].T

    def update(self, **kwargs):
        """Update the option"""

        # reset the impacts and the costs
        self.reset()

        # update the parameters of the options
        self._parameters.update(kwargs)

        # update the new start
        self._start = self._parameters.get('start', 0)

        # update the values and costs
        self.set_values(**self._parameters)
        self.set_costs(**self._parameters)


class OptionDemand(BaseOption):
    """Option to reduce the demand."""


class OptionLeakage(OptionDemand):
    """Option to reduce the leakage."""


class OptionPrivate(OptionDemand):
    """Option to reduce the private demand."""


class OptionSupply(BaseOption):
    """Option to increase the supply"""


class OptionCapacity(BaseOption):
    """Option to increase the capacity"""


class OptionMisc(BaseOption):
    """Miscellaneous option."""


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
