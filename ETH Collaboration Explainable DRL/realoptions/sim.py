# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : sim.py -- An other area to simulate stuff
# Author    : Jürgen Hackl <hackl.research@gmail.com>
# Creation  : 2021-01-21
# Time-stamp: <Mon 2023-01-09 15:13 juergen>
#
# Copyright (c) 2021 Jürgen Hackl <hackl.research@gmail.com>
# =============================================================================
import numpy as np  # type: ignore
from realoptions import (
    OrnsteinUhlenbeckProcess, DeterministicProcess, MarkedPoissonPointProcess,
    NormalProcess, WaterPrivate, WaterLeakage, WaterService, WaterIndustry,
    WaterExternal, WaterStorage, OptionSupply, OptionLeakage, OptionPrivate,
    OptionMisc, OptionCapacity, WaterSupply, plot_d_and_s, plot_costs,
    plot_risk, plot_input, plot_time, plot_risk_comp, cost_benefit,
    plot_nb_comp
)

# General variables
# =================
SIMULATIONS = 2000  # number of simulations per option
C_ENV = 20  # environmental costs by exceeding the fish pass level [m£/year]
C_SUP = 100  # environmental costs by exceeding the supply level [m£/year]
C_WR = 100  # costs if demand can not be fulfilled [m£/year]
RATE = 0.035  # discount rate [-]
DISCOUNT = True  # discounting is enabled [True / False]
DYNAMIC = True  # consider dynamic allocation i.e. option 4 [True / False]


def generate_input_models():
    """Function to generate the models for the inputs"""
    # population
    # - - - - -
    # low
    pop_model = OrnsteinUhlenbeckProcess(100000, 3000, .03, 64350)
    # medium
    # pop_model = OrnsteinUhlenbeckProcess(110000, 3000, .05, 64350)
    # high
    # pop_model = OrnsteinUhlenbeckProcess(120000, 10000, .07, 64350)

    # industry
    ind_model = OrnsteinUhlenbeckProcess(1.2, .05, .03, .64)

    # per capital consumption
    pcc_model = DeterministicProcess(175)

    # people per household
    pph_model = DeterministicProcess(2.152)

    # external events
    eev_model = MarkedPoissonPointProcess(
        1/20, np.random.poisson, parameters={'lam': 30})

    # external demand
    edm_model = DeterministicProcess(18)  # [Ml/d]

    # rainfall data
    pre_model = NormalProcess(1430, 140, mu_shift=-.001, sigma_shift=.005)

    # dict with input models
    return {
        'pop': pop_model,
        'pcc': pcc_model,
        'pph': pph_model,
        'ind': ind_model,
        'eev': eev_model,
        'edm': edm_model,
        'pre': pre_model
    }


def generate_consumers(processes=None):
    """Function to generate all consumers in Inverness"""

    private = WaterPrivate(
        name='private water consumption',
        priority=100,
        scale_factor=1.1,
        compensation=C_WR,
        processes=processes
    )

    leakage = WaterLeakage(
        name='water leakage',
        scale_factor=0.3,
        processes=processes
    )

    service = WaterService(
        name='service water consumption',
        priority=500,
        scale_factor=0.025,
        compensation=C_WR,
        processes=processes
    )

    industry = WaterIndustry(
        name='industrial water consumption',
        priority=400,
        compensation=C_WR,
        base_demand=0.5,
        scale_factor=1.,
        processes=processes
    )

    external = WaterExternal(
        name='external demand',
        priority=50,
        compensation=C_WR,
        processes=processes
    )

    return [private, leakage, service, industry, external]


def generate_producers(processes=None):
    """Function to generate all consumers in Inverness"""

    duntelchaig = WaterStorage(
        name='Reservoir A',  # 'Loch Duntelchaig',
        storage=144402,  # [Ml]
        operation=7373,  # [Ml]
        intake=13646,  # [Ml]
        area=5546219,  # [m2]
        dynamic=True,
        processes=processes,
    )

    ashie = WaterStorage(
        name='Reservoir B',  # 'Loch Ashie',
        operation=3.7 * 365,  # [Ml/d * d]
        active=False,
    )

    return [duntelchaig, ashie]


def generate_options(discount_rate=0.0354):
    """Function to generate a list of options"""

    # Base option
    # -----------
    base = OptionSupply(
        name='Base costs',
        increase=0,
        enhancement=0,
        replacement=1.475,
        opex=.145,
        duration=0,
        start=0,
        discount_rate=discount_rate,
    )

    # Options to reduce the demand
    # ----------------------------
    leakage = OptionLeakage(
        name='Leakage reduction',
        reduction=2,
        start=0,
        duration=2,
        enhancement=.5,
        replacement=.03,
        opex=.015,
        discount_rate=discount_rate,
    )

    efficiency = OptionPrivate(
        name='Campaign for PCC reduction',
        reduction=1,
        start=0,
        duration=0,
        enhancement=.0,
        replacement=.0375,
        opex=0,
        discount_rate=discount_rate,
    )

    metering = OptionPrivate(
        name='Customer metering for PCC reduction',
        reduction=4,
        start=0,
        duration=3,
        enhancement=8.75,
        replacement=.75,
        opex=0.0,
        discount_rate=discount_rate,
    )

    # Options to increase supply
    # --------------------------
    link = OptionSupply(
        name='Assynt WTW link',
        increase=5,
        start=0,
        duration=2,
        enhancement=28.5,
        replacement=.07,
        opex=0,
        discount_rate=discount_rate,
    )

    raw_20 = OptionSupply(
        name='New 20Ml/d raw water source',
        dynamic=False,
        increase=20,
        start=0,
        duration=2,
        enhancement=34.2,
        replacement=0.0,
        opex=0.5,
        discount_rate=discount_rate,
    )

    raw_40 = OptionSupply(
        name='New 40Ml/d raw water source',
        increase=40,
        start=0,
        duration=2,
        enhancement=38.23,
        replacement=0.0,
        opex=0.5,
        discount_rate=discount_rate,
    )

    raw_20_30 = OptionSupply(
        name='New 20Ml/d raw water source in year 30',
        increase=20,
        start=30,
        duration=2,
        enhancement=34.2,
        replacement=0.4,
        opex=0.1,
        discount_rate=discount_rate,
    )

    raw_20_50 = OptionSupply(
        name='New 20Ml/d raw water source in year 50',
        increase=20,
        start=50,
        duration=2,
        enhancement=34.2,
        replacement=.4,
        opex=0.1,
        discount_rate=discount_rate,
    )

    raw_40_init_20 = OptionSupply(
        name='New 40Ml/d raw water source initial at 20Ml/d',
        increase=40,
        start=0,
        duration=2,
        enhancement=36.43,
        replacement=0.,
        opex=0.5,
        discount_rate=discount_rate,
    )

    raw_40_init_20_10 = OptionSupply(
        name='New 40Ml/d raw water source initial at 20Ml/d in year 10',
        increase=40,
        start=10,
        duration=2,
        enhancement=36.43,
        replacement=.0,
        opex=0.5,
        discount_rate=discount_rate,
    )

    new_wtw = OptionSupply(
        name='New source &  WTW',
        increase=15,
        enhancement=50.9,
        replacement=0.800,
        opex=.145,
        duration=2,
        start=0,
        discount_rate=discount_rate,
    )

    # Additional costs for maintenance
    # --------------------------------
    meter = OptionMisc(
        name='Meter replacement',
        start=3,
        duration=2,
        enhancement=0.1,
        interval=10,
        discount_rate=discount_rate,
    )

    maintenance_10 = OptionMisc(
        name='10 year maintenance',
        start=11,
        replacement=1.422,
        interval=(None, 10, None),
        discount_rate=discount_rate,
    )

    maintenance_10_10 = OptionMisc(
        name='10 year maintenance',
        start=21,
        replacement=1.422,
        interval=(None, 10, None),
        discount_rate=discount_rate,
    )

    meica = OptionMisc(
        name='25 year MEICA replacement',
        start=27,
        replacement=10.82,
        interval=(None, 25, None),
        discount_rate=discount_rate,
    )

    # Options to increase capacity
    # ----------------------------
    link_capacity = OptionCapacity(
        name='Assynt WTW capacity',
        increase=5,
        start=0,
        duration=2,
        discount_rate=discount_rate,
    )

    capacity = OptionCapacity(
        name='10Ml/d WTW capacity increase',
        dynamic=False,
        increase=10,
        start=43,
        duration=1,
        enhancement=15,
        replacement=0,
        opex=0,
        discount_rate=discount_rate,
    )

    new_capacity = OptionCapacity(
        name='New WTW capacity',
        increase=15,
        start=0,
        duration=2,
        enhancement=0,
        replacement=0,
        opex=0,
        discount_rate=discount_rate,
    )

    raw_20_dyn = OptionSupply(
        name='New 20Ml/d raw water source',
        dynamic=True,
        increase=20,
        start=0,
        duration=2,
        enhancement=34.2,
        replacement=0.0,
        opex=0.5,
        discount_rate=discount_rate,
    )

    cap_dyn = OptionCapacity(
        name='10Ml/d WTW capacity increase',
        dynamic=True,
        increase=10,
        start=43,
        duration=1,
        enhancement=15,
        replacement=0,
        opex=0,
        discount_rate=discount_rate,
    )

    # Generate Option combinations
    # ----------------------------

    options = {}
    options['0'] = [base]
    options['1a'] = [base, leakage]
    options['1b'] = [base, leakage, efficiency]
    options['1c'] = [base, leakage, metering, meter]
    options['1d'] = [base, leakage, link, link_capacity]
    options['2a'] = [base, raw_20, maintenance_10, meica]
    options['2b'] = [base, raw_40, capacity, maintenance_10, meica]
    options['2c'] = [base, raw_20, raw_20_30, capacity, maintenance_10, meica]
    options['2d'] = [base, raw_20, raw_20_50, capacity, maintenance_10, meica]
    options['2e'] = [base, raw_40_init_20, capacity, maintenance_10, meica]
    options['2f'] = [base, raw_40_init_20_10,
                     capacity, maintenance_10_10, meica]
    options['3'] = [base, new_wtw, new_capacity,
                    raw_20_50, maintenance_10, meica]

    # dynamic option
    options['4'] = [base, cap_dyn, raw_20_dyn]

    return options


def scenario(simulations=100, discount_rate=0.035,
             options=None, discounted=True, optimization=True):
    """Main function of the program"""

    # variables to store the outputs
    topics = ['input', 'cost', 'compensation', 'investment',
              'demand', 'supply', 'level', 'capacity', 'timing']
    result = {topic: {'data': [], 'labels': []} for topic in topics}

    # random input models
    models = generate_input_models()

    # generate consumers
    consumers = generate_consumers()

    # generate producers
    producers = generate_producers()

    # generate wtw
    wtw = WaterSupply(name='Inverness WTW', capacity=38.5)  # [Ml/d]

    # combine all models
    all_models = consumers + producers + [wtw]

    # discount vector
    discount = np.ones((1, 61)) / (1+discount_rate) ** np.arange(0, 61, step=1)

    # run simulations for the scenario
    for i in range(simulations):

        # Calculations
        # ============
        # simulation of random variables
        processes = {key: model.simulate()for key, model in models.items()}

        if optimization:
            temp_options = []
            for option in options:
                if not option.dynamic:
                    temp_options.append(option)
                else:
                    opt_d = option._parameters['duration']
                    opt_t = len(option._time)
                    for t in range(opt_t-opt_d):
                        option.update(start=t)

                        # reset all models
                        for model in all_models:
                            model.reset(processes=processes)

                        # balance supply and demand
                        wtw.run(producers, consumers, options=temp_options)

                        # check if iteration should be stopped
                        if isinstance(option, OptionCapacity):
                            if wtw.unsupplied[t+opt_d] > 0:
                                temp_options.append(option)
                                break

                        if isinstance(option, OptionSupply):
                            level = producers[0].failure()
                            if level[t+opt_d] == 1:
                                temp_options.append(option)
                                break

        # reset all models
        for model in all_models:
            model.reset(processes=processes)

        # balance supply and demand
        supply, demand = wtw.run(producers, consumers, options=options)

        # Costs
        # =====
        # cost of inadequate water supply
        c_iws = {'data': [], 'labels': []}
        for consumer in consumers:
            c_iws['data'].append(consumer.costs(discount_rate=discount_rate,
                                                discounted=discounted))
            c_iws['labels'].append('Costs of %s' % consumer.name)

        # cost of interventions
        c_int = {'data': [], 'labels': []}
        for option in options:
            c_int['data'].append(option.costs(discounted=discounted))
            c_int['labels'].append(option.name)

        # cost of environmental impact
        c_env = {'data': [], 'labels': []}
        for lake in producers:
            if lake.dynamic:
                c_env['data'].append(lake.failure('operation') * C_ENV *
                                     (discount.T if discounted else 1))
                c_env['labels'].append('Costs of fish passing')

        # cost for under-pumping
        c_sup = {'data': [], 'labels': []}
        for lake in producers:
            if lake.dynamic:
                c_sup['data'].append(lake.failure('intake') * C_SUP *
                                     (discount.T if discounted else 1))
                c_sup['labels'].append('Costs of under-pumping')

        # Store results
        # =============
        result['input']['data'].append(
            np.hstack([m for m in processes.values()]))
        result['input']['labels'] = list(processes.keys())
        result['supply']['data'].append(supply['data'])
        result['supply']['labels'] = supply['labels']
        result['demand']['data'].append(demand['data'])
        result['demand']['labels'] = demand['labels']
        result['level']['data'].append(producers[0].storage)
        result['level']['labels'] = ['Water level']
        result['capacity']['data'].append(wtw.capacity())
        result['capacity']['labels'] = ['WTW capacity']

        result['investment']['data'].append(np.hstack(c_int['data']))
        result['investment']['labels'] = c_int['labels']

        _d,  _l = _combine(c_iws, c_env, c_sup)
        result['compensation']['data'].append(np.hstack(_d))
        result['compensation']['labels'] = _l

        c_wr = {'data': [np.sum(np.hstack(c_iws['data']),
                                axis=1, keepdims=True)],
                'labels': ['Cost for inadequate water supply']}

        c_inv = {'data': [np.sum(np.hstack(c_int['data']),
                                 axis=1, keepdims=True)],
                 'labels': ['Cost for interventions']}

        _d,  _l = _combine(c_wr, c_env, c_sup, c_inv)
        result['cost']['data'].append(np.hstack(_d))
        result['cost']['labels'] = _l

        _d = np.zeros((61, len(options)))
        for j, opt in enumerate(options):
            _d[opt.start][j] = 1

        result['timing']['data'].append(_d)
        result['timing']['labels'] = [o.name for o in options]

    # Format result
    # =============
    for topic in topics:
        result[topic]['data'] = np.dstack(result[topic]['data'])
        # print(topic, result[topic]['data'].shape)

    return result


def _combine(*args):
    """Helper function to combine different costs"""
    data = []
    labels = []
    for model in args:
        data.extend([*model['data']])
        labels.extend(model['labels'])
    return (data, labels)


def single_scenario(option, name='', filename='', fformat='pdf'):
    """Calculate results for a single option"""
    # calculate single scenario
    result = scenario(simulations=SIMULATIONS,
                      discount_rate=RATE,
                      options=option,
                      discounted=DISCOUNT)

    # visualise the results
    # plot_d_and_s(result, name=name, filename=filename, fformat=fformat)
    # plot_costs(result, name=name)
    # plot_risk(result, name=name)
    # plot_input(result, name=name)
    if DYNAMIC:
        plot_time(result, name=name)


def multiple_scenarios(options):
    """Calculate results for multiple option"""
    # dict to store results
    results = {}

    # calculate results for all options
    for key, opt in options.items():
        # only calculate dynamic option if enabled
        if key == '4' and not DYNAMIC:
            break

        # calculate scenario
        results[key] = scenario(simulations=SIMULATIONS,
                                discount_rate=RATE,
                                options=opt,
                                discounted=DISCOUNT)

        # plot options for single result
    #     filename = './plots/option-{}'.format(key)
    #     name = 'Opition {}'.format(key)
    #     fformat = 'png'

    #     # plot single results
    #     plot_d_and_s(results[key], name=name,
    #                  filename=filename+'_s&d', fformat=fformat)
    #     plot_costs(results[key], name=name,
    #                filename=filename+'_cost', fformat=fformat)
    #     plot_risk(results[key], name=name,
    #               filename=filename+'_risk', fformat=fformat)
    #     plot_input(results[key], name=name,
    #                filename=filename+'_input', fformat=fformat)

    # # Plot all results together
    # plot_risk_comp(results, name='', fformat='pdf')
    # plot_nb_comp(results, name='', fformat='pdf')
    # cost_benefit(results, base='0')


def multiple_scenarios_res(options):
    """Calculate results for multiple option"""
    # dict to store results
    results = {}

    # calculate results for all options
    for key, opt in options.items():
        # only calculate dynamic option if enabled
        if key == '4' and not DYNAMIC:
            break

        # calculate scenario
        results[key] = scenario(simulations=SIMULATIONS,
                                discount_rate=RATE,
                                options=opt,
                                discounted=DISCOUNT)

        # plot options for single result
    #     filename = './plots/option-{}'.format(key)
    #     name = 'Opition {}'.format(key)
    #     fformat = 'png'

    #     # plot single results
    #     plot_d_and_s(results[key], name=name,
    #                  filename=filename+'_s&d', fformat=fformat)
    #     plot_costs(results[key], name=name,
    #                filename=filename+'_cost', fformat=fformat)
    #     plot_risk(results[key], name=name,
    #               filename=filename+'_risk', fformat=fformat)
    #     plot_input(results[key], name=name,
    #                filename=filename+'_input', fformat=fformat)

    # # Plot all results together
    # plot_risk_comp(results, name='', fformat='pdf')
    # plot_nb_comp(results, name='', fformat='pdf')
    # cost_benefit(results, base='0')
    return results

def main():
    """Main function of the program"""

    # get options
    options = generate_options(RATE)

    # calculate single scenario
    # single_scenario(options['4'], name='Option 4',
    #                 filename='option-4_time', fformat='pdf')

    # calculate multiple scenarios
    #multiple_scenarios(options)
    results = multiple_scenarios_res(options)


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
