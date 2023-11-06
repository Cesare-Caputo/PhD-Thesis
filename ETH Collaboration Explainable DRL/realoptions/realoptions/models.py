"""Models for the evaluation"""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : models.py -- some models
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2020-12-17
# Time-stamp: <Fri 2021-02-19 17:53 juergen>
#
# Copyright (c) 2020 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
from typing import Optional, Any
import numpy as np  # type: ignore
from realoptions.options import (OptionLeakage, OptionPrivate,
                                 OptionCapacity, OptionSupply)


class BaseModel:
    """Class to make models"""

    def __init__(self, name: Optional[str] = None,
                 priority: int = 1000,
                 compensation: float = 0.0,
                 processes: Optional[dict] = None,
                 start_year: float = 0.0,
                 end_year: float = 61.0,
                 dt: float = 1.0,
                 **kwargs: Any) -> None:
        """Initialize base option"""
        # pylint: disable=too-many-arguments

        # Variables
        # name of the model
        self._name = name

        # priority of the model
        self._priority = priority

        # compensation for insufficient service
        self._compensation = compensation

        # collection of stochastic processes
        self._processes: dict = {}

        # store the parameters
        self._parameters = kwargs

        # Initialize matrices
        # time vector
        self._time = np.arange(start_year, end_year, step=dt)

        # TODO: make the supply and demand  more flexible
        # supply and demand of the model
        # 0 - supply
        # 1 - demand
        self._values = np.zeros((2, len(self)))

        # add processes if given
        if processes and not isinstance(processes, dict):
            # if no dict is given add the process as default
            self._processes['default'] = processes
        elif isinstance(processes, dict):
            # otherwise update the dict
            self._processes.update(processes)

    def __len__(self) -> int:
        """Return the lengt of the model in terms of columns"""
        return len(self._time)

    @property
    def name(self) -> Optional[str]:
        """Return the name of the model."""
        return self._name

    @property
    def priority(self) -> int:
        """Return the priority of the model."""
        return self._priority

    @property
    def processes(self) -> dict:
        """Return a dict of all processes."""
        return self._processes

    @property
    def compensation(self) -> float:
        """Return the unit costs for compensation."""
        return self._compensation

    def supply(self, *options, **kwargs):
        """Return the supply of the model"""
        # pylint: disable=unused-argument
        return self._values[[0]].T

    def demand(self, *options, **kwargs):
        """Return the demand of the model"""
        # pylint: disable=unused-argument
        return self._values[[1]].T

    def reset(self, **kwargs) -> None:
        """Reset the model."""
        processes = kwargs.get('processes', {})
        self._processes = processes
        self._values = np.zeros((2, len(self)))

    def _zero(self, columns: int = 1, reverse=False):
        """Helper function to generate zero matrix"""
        if reverse:
            zero = np.zeros((columns, len(self)))
        else:
            zero = np.zeros((len(self), columns))
        return zero


class BaseDemandModel(BaseModel):
    """Model for private water usage."""

    def __init__(self, base_demand: float = 0.0,
                 scale_factor: float = 1.0,
                 **kwargs) -> None:
        # intialize base class
        super().__init__(**kwargs)

        # constant for the base demand
        self.base_demand = base_demand

        # factor to scale the demand
        self.scale_factor = scale_factor

    def supply(self, *options, **kwargs):
        """Return the supply of the model"""
        # pylint: disable=unused-argument
        # get the demand
        provided = kwargs.get('provided', None)
        if provided is not None:
            self._values[[0]] = provided.T

        return super().supply(*options, **kwargs)

    def costs(self, discount_rate: float = 0.0, discounted: bool = False):
        """Function to return the compensation costs for insufficient supply"""
        cost = np.abs(self._values[[0]].T - self._values[[1]].T)
        cost *= self.compensation

        if discounted:
            cost *= (np.ones((1, len(self))) /
                     (1+discount_rate) ** self._time).T
        return cost


class WaterPrivate(BaseDemandModel):
    """Model for private water usage."""

    def demand(self, *options, **kwargs):
        """Calculate the demand"""

        # population
        pop = self.processes['pop']  # [p]

        # per capital consumption
        pcc = self.processes['pcc'] * 3.5  # [l/h/d]

        # people per household
        pph = self.processes['pph']  # [p/h]

        reduction = 0
        for option in options:
            if isinstance(option, OptionPrivate):
                reduction = option.reduction()

        demand = (pop/pph*pcc/10**6) * self.scale_factor + \
            self.base_demand - reduction  # [Ml/d]

        # update values
        self._values[1] = demand.T
        return demand


class WaterLeakage(BaseDemandModel):
    """Model for water leakage. """

    def demand(self, *options, **kwargs):
        """Calculate the demand"""

        # get random processes
        pop = self.processes['pop']  # [Ml/d]
        pcc = self.processes['pcc']  # [l/h/d]
        pph = self.processes['pph']  # [p/h]

        # get reductions
        reduction = 0
        for option in options:
            if isinstance(option, OptionLeakage):
                reduction = option.reduction()

        # calculate demand
        demand = pop/pph*pcc/10**6 * self.scale_factor + \
            self.base_demand - reduction  # [Ml/d]

        # update demand
        self._values[1] = demand.T
        return demand


class WaterService(BaseDemandModel):
    """Model for service operators."""

    def demand(self, *options, **kwargs):
        """Calculate the demand"""

        # get random processes
        pop = self.processes['pop']  # [Ml/d]
        pcc = self.processes['pcc']  # [l/h/d]
        pph = self.processes['pph']  # [p/h]

        # get reductions
        reduction = 0

        # calculate demand
        demand = pop/pph*pcc/10**6 * self.scale_factor + \
            self.base_demand - reduction  # [Ml/d]

        # update demand
        self._values[1] = demand.T
        return demand


class WaterIndustry(BaseDemandModel):
    """Model for industrial water usage."""

    def demand(self, *options, **kwargs):
        """Calculate the demand"""

        # get random processes
        ind = self.processes['ind']  # [Ml/d]

        # get reductions
        reduction = 0

        # calculate demand
        demand = ind * self.scale_factor + \
            self.base_demand - reduction  # [Ml/d]

        # update demand
        self._values[1] = demand.T
        return demand


class WaterExternal(BaseDemandModel):
    """Model for external water usage."""

    def demand(self, *options, **kwargs):
        """Calculate the demand"""

        # get random processes
        eev = self.processes['eev']  # [d/y]
        edm = self.processes['edm']  # [Ml/d]

        # get reductions
        reduction = 0

        # calculate demand
        demand = eev * edm / 365 * self.scale_factor + \
            self.base_demand - reduction  # [Ml/d]

        # update demand
        self._values[1] = demand.T
        return demand


class WaterStorage(BaseModel):
    """Documentation for WaterStorage"""

    def __init__(self, storage: float = float('inf'),
                 operation: float = 0.0,
                 intake: float = 0.0,
                 area: float = 0.0,
                 dynamic: bool = False,
                 **kwargs) -> None:
        """Initialize the water storage"""
        # pylint: disable=too-many-arguments

        # intialize base class
        super().__init__(**kwargs)

        #  storage
        self._storage = storage  # [Ml]

        # maximum operation (e.g. fish passing)
        self._operation = operation  # [Ml]

        # maximum intake
        self._intake = intake  # [Ml]

        # area (OSM)
        self._area = area  # [m2]

        # if the storage can be used dynamically
        # i.e. the outtake may vary due to optimization
        self._dynamic = dynamic

        # water storage over time
        self.storage = np.full((len(self), 1), self._storage)  # [Ml]

        # estimate initial supply
        self._values[[0]] = self._operation/365  # [Ml/d]

        # matrix for detailed usage of the storage
        # 0 - normal usage
        # 1 - availabe to refill storage
        # 2 - under pumping
        self._usage = np.zeros((3, len(self)))

    @property
    def dynamic(self) -> bool:
        """Return if the storage can be used dynamically."""
        return self._dynamic

    @property
    def operation_level(self) -> float:
        """Return the operatinal water level for the storage."""
        return self._storage - self._operation  # [Ml]

    @property
    def intake_level(self) -> float:
        """Return the intake water level for the storage."""
        return self._storage - self._intake  # [Ml]

    def failure(self, failure: str = 'operation'):
        """Return the operatinal failures."""
        if failure == 'operation':
            level = self.storage-self.operation_level
        else:
            level = self.storage-self.intake_level
        # convert level in binary vector 0 okay, 1 failure
        level[level > 0] = 0
        level[level < 0] = 1

        return level

    def reset(self, **kwargs) -> None:
        """Reset the model."""
        processes = kwargs.get('processes', {})
        self._processes = processes
        self._values = np.zeros((2, len(self)))
        self.storage = np.full((len(self), 1), self._storage)  # [Ml]
        self._values[[0]] = self._operation/365  # [Ml/d]
        self._usage = np.zeros((3, len(self)))

    def inflow(self, year=False, **kwargs):
        """Calculate inflow"""
        # pylint: disable=unused-argument

        # precipitation
        pre = self.processes.get('pre', self._zero())  # [mm/m2/a] -> [l/m2/a]

        inflow = pre * self._area / 10**6 / 365

        # get the inflow per year
        if year:
            inflow *= 365

        return inflow  # / 365  # [Ml/d]

    def supply(self, *options, **kwargs):
        """Return the supply of the model"""
        # pylint: disable=unused-argument
        # get the demand
        demand = kwargs.get('demand', None)
        if self.dynamic and demand is not None:

            inflow = self.inflow().flatten()
            demand = demand.flatten()
            delta = inflow - demand

            self._usage[0][delta > 0] = demand[delta > 0]
            self._usage[0][delta <= 0] = inflow[delta <= 0]
            self._usage[1][delta > 0] = delta[delta > 0]
            self._usage[2][delta <= 0] = delta[delta <= 0] * -1

            self._values[[1]] = demand
            self._values[[0]] = inflow
            supply = self._usage.T

            self.update_storage(delta)
        elif self.dynamic and demand is None:
            supply = self.inflow()
            self._values[[0]] = supply.T
        else:
            supply = super().supply()

        return supply

    def update_storage(self, delta) -> None:
        """Update the water content in the storage"""
        delta *= 365
        for time, change in enumerate(delta):
            if time == 0:
                value = self._storage + change
            else:
                value = self.storage[time-1] + change

            if value > self._storage:
                value = self._storage
            elif value < 0:
                value = 0

            self.storage[time] = value


class WaterSupply(BaseModel):
    """Documentation for WaterSupply"""

    def __init__(self, capacity: float = float('inf'), **kwargs) -> None:
        # intialize base class
        super().__init__(**kwargs)

        # max capacity
        self._max_capacity = capacity

        # unsupplied demand
        self.unsupplied = 0.0

        # capacity of the WTW
        self._capacity = np.full((len(self), 1), capacity)  # [Ml/d]

    def reset(self, **kwargs) -> None:
        """Reset the model."""
        self._capacity = np.full((len(self), 1), self._max_capacity)  # [Ml/d]
        self.unsupplied = 0.0

    def capacity(self, *options, **kwargs):
        """Get the capacity of the WTW"""
        # pylint: disable=unused-argument

        increase = 0
        for option in options:
            if isinstance(option, OptionCapacity):
                increase += option.increase()

        self._capacity += increase

        return self._capacity

    def run(self, producers, consumers, options=None):
        """Balance supply and demand"""
        # pylint: disable=too-many-locals

        # temp variables to store the results
        supply = []
        demand = []
        s_labels = []
        d_labels = []

        # find active and inactive  producers
        active = [p for p in producers if p.dynamic]
        inactive = [p for p in producers if not p.dynamic]

        # calculate base supply
        for producer in inactive:
            supply.append(producer.supply())
            s_labels.append(producer.name)

        # account for options which increase the supply
        for option in options:
            if isinstance(option, OptionSupply):
                increase = option.increase()
                if np.sum(increase) > 0:
                    supply.append(increase)
                    s_labels.append(option.name)

        # combine base supply
        supply = np.hstack(supply)

        # Sort consumers based on their priorities
        consumers = [x for _, x in sorted(
            zip([c.priority for c in consumers], consumers),
            key=lambda pair: pair[0], reverse=False)]

        # calculate demands
        demand = np.hstack([c.demand(*options) for c in consumers])
        d_labels = [c.name for c in consumers]

        # get capacity of the wtw
        capacity = self.capacity(*options)

        # total demand and supply
        total_demand = np.sum(demand, axis=1, keepdims=True)
        total_supply = np.sum(supply, axis=1, keepdims=True)

        # demand able to be supplied
        supplied = np.clip(total_demand, 0, capacity)

        # demand not able to be supplied
        unsupplied = total_demand-supplied
        self.unsupplied = unsupplied.copy()

        # delta between demand and supply which can be provided
        delta = total_supply - supplied

        # additional supply extracted form an dynamic storage
        additional = delta.copy() * -1
        additional[additional < 0] = 0

        # supply from an dynamic storage
        for producer in active:
            _supply = producer.supply(demand=additional/len(active))
            supply = np.hstack([supply, _supply])
            s_labels.append(producer.name)
            s_labels.append(producer.name + ' in storage')
            s_labels.append(producer.name + ' under-pumping')

        # initialize variables
        delta = np.zeros((len(self), len(consumers)))

        # find unsupplied demand per consumer
        for i in range(len(consumers)):
            unsupplied -= demand[:, [i]]
            delta[:, [i]] = unsupplied

        delta[delta > 0] = 0
        delta = demand+delta
        delta[delta < 0] = 0

        provided = demand - delta

        # update the supply provided
        for i, consumer in enumerate(consumers):
            consumer.supply(provided=provided[:, [[i]]])

        return ({'data': supply, 'labels': s_labels},
                {'data': demand, 'labels': d_labels})

    def run_rl(self, producers, consumers):
        """Balance supply and demand"""
        # pylint: disable=too-many-locals

        # temp variables to store the results
        supply = []
        demand = []
        s_labels = []
        d_labels = []

        # find active and inactive  producers
        active = [p for p in producers if p.dynamic]
        inactive = [p for p in producers if not p.dynamic]

        # calculate base supply
        for producer in inactive:
            supply.append(producer.supply())
            s_labels.append(producer.name)

        # combine base supply
        supply = np.hstack(supply)

        # Sort consumers based on their priorities
        consumers = [x for _, x in sorted(
            zip([c.priority for c in consumers], consumers),
            key=lambda pair: pair[0], reverse=False)]

        # calculate demands
        demand = np.hstack([c.demand() for c in consumers])
        d_labels = [c.name for c in consumers]

        # get capacity of the wtw
        capacity = self.capacity()

        # total demand and supply
        total_demand = np.sum(demand, axis=1, keepdims=True)
        total_supply = np.sum(supply, axis=1, keepdims=True)

        # demand able to be supplied
        supplied = np.clip(total_demand, 0, capacity)
        # demand not able to be supplied
        unsupplied = total_demand-supplied
        self.unsupplied = unsupplied.copy()

        # delta between demand and supply which can be provided
        delta = total_supply - supplied

        # additional supply extracted form an dynamic storage
        additional = delta.copy() * -1
        additional[additional < 0] = 0

        # supply from an dynamic storage
        for producer in active:
            _supply = producer.supply(demand=additional/len(active))
            supply = np.hstack([supply, _supply])
            s_labels.append(producer.name)
            s_labels.append(producer.name + ' in storage')
            s_labels.append(producer.name + ' under-pumping')

        # initialize variables
        delta = np.zeros((len(self), len(consumers)))

        # find unsupplied demand per consumer
        for i in range(len(consumers)):
            unsupplied -= demand[:, [i]]
            delta[:, [i]] = unsupplied

        delta[delta > 0] = 0
        delta = demand+delta
        delta[delta < 0] = 0

        provided = demand - delta

        # update the supply provided
        for i, consumer in enumerate(consumers):
            consumer.supply(provided=provided[:, [[i]]])

        return ({'data': supply, 'labels': s_labels},
                {'data': demand, 'labels': d_labels})






# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
