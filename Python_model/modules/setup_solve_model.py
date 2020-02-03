"""
oemof application for research project quarree100.

Based on the excel_reader example of oemof_examples repository:
https://github.com/oemof/oemof-examples

see also: https://github.com/quarree100/q100_oemof_app

Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""


from oemof.tools import logger
from oemof.tools import economics
import oemof.solph as solph
import logging
import pandas as pd
import numpy as np


def nodes_from_excel(filename):

    xls = pd.ExcelFile(filename)

    nodes_data = {'buses': xls.parse('Buses'),
                  'commodity_sources': xls.parse('Sources'),
                  'sources_series': xls.parse('Sources_series'),
                  'demand': xls.parse('Demand'),
                  'sinks': xls.parse('Sinks'),
                  'transformer': xls.parse('Transformer'),
                  'storages': xls.parse('Storages'),
                  'timeseries': xls.parse('Timeseries'),
                  'general': xls.parse('General')
                  }

    # set datetime index
    nodes_data['timeseries'].set_index('timestamp', inplace=True)
    nodes_data['timeseries'].index = pd.to_datetime(
        nodes_data['timeseries'].index)

    print('Data from Excel file {} imported.'
          .format(filename))

    return nodes_data


def create_nodes(nd=None):
    """Create nodes (oemof objects) from node dict

    Parameters
    ----------
    nd : :obj:`dict`
        Nodes data

    Returns
    -------
    nodes : `obj`:dict of :class:`nodes <oemof.network.Node>`

    TODO: insert a test check for all data-timeseries readers if label is in timeseries!
    """

    if not nd:
        raise ValueError('No nodes data provided.')

    nodes = []

    # Create Bus objects from buses table
    busd = {}

    for i, b in nd['buses'].iterrows():
        if b['active']:
            bus = solph.Bus(label=b['label'])
            nodes.append(bus)

            busd[b['label']] = bus
            if b['excess']:
                nodes.append(
                    solph.Sink(label=b['label'] + '_excess',
                               inputs={busd[b['label']]: solph.Flow(
                                   variable_costs=b['excess costs'])})
                )
            if b['shortage']:
                nodes.append(
                    solph.Source(label=b['label'] + '_shortage',
                                 outputs={busd[b['label']]: solph.Flow(
                                     variable_costs=b['shortage costs'])})
                    )

    # Create Source objects from table 'Sources'
    for i, cs in nd['commodity_sources'].iterrows():
        if cs['active']:

            outflow_args = {}

            if cs['cost_series']:
                for col in nd['timeseries'].columns.values:
                    if col.split('..')[0] == cs['label']:
                        outflow_args['variable_costs'] = nd['timeseries'][col]
            else:
                outflow_args['variable_costs'] = cs['variable costs']

            if cs['emission_series']:
                for col in nd['timeseries'].columns.values:
                    if col.split('.')[0] == cs['label']:
                        outflow_args['emission_factor'] = nd['timeseries'][col]
            else:
                outflow_args['emission_factor'] = \
                    np.full(nd['general']['timesteps'][0], cs['emissions'])

            nodes.append(
                solph.Source(
                    label=cs['label'],
                    outputs={busd[cs['to']]: solph.Flow(**outflow_args)})
                )

    # Create Source objects with fixed time series from 'renewables' table
    for i, ss in nd['sources_series'].iterrows():
        if ss['active']:
            if ss['invest']:
                # calculation epc
                epc = economics.annuity(
                    capex=ss['capex'], n=ss['n'],
                    wacc=nd['general']['interest rate'][0]) * nd[
                          'general']['timesteps'][0] / 8760

                # get time series for node and parameter
                for col in nd['timeseries'].columns.values:
                    if col.split('.')[0] == ss['label']:
                        av = nd['timeseries'][col]

                # create
                nodes.append(
                    solph.Source(label=ss['label'],
                                 outputs={
                                     busd[ss['to']]: solph.Flow(
                                         fixed=True,
                                         actual_value=av,
                                         investment=solph.Investment(
                                             ep_costs=epc,
                                             maximum=ss['max_invest']))}))

            else:
                # set static outflow values
                outflow_args = {'nominal_value': ss['installed'],
                                'fixed': True}
                # get time series for node and parameter
                for col in nd['timeseries'].columns.values:
                    if col.split('.')[0] == ss['label']:
                        outflow_args[col.split('.')[1]] = nd['timeseries'][col]

                # create
                nodes.append(
                    solph.Source(label=ss['label'],
                                 outputs={
                                     busd[ss['to']]: solph.Flow(**outflow_args)})
                )

    # Create Sink objects with fixed time series from 'demand' table
    for i, de in nd['demand'].iterrows():
        if de['active']:
            # set static inflow values
            inflow_args = {'nominal_value': de['scalingfactor'],
                           'fixed': de['fixed']}
            # get time series for node and parameter
            for col in nd['timeseries'].columns.values:
                if col.split('.')[0] == de['label']:
                    inflow_args[col.split('.')[1]] = nd['timeseries'][col]

            # create
            nodes.append(
                solph.Sink(label=de['label'],
                           inputs={
                               busd[de['from']]: solph.Flow(**inflow_args)})
            )

    # Create further sink objects
    for i, sk in nd['sinks'].iterrows():
        if sk['active']:

            outflow_args = {'nominal_value': sk['p_max'],
                            'summed_max': sk['total_max']}

            if sk['cost_series']:
                for col in nd['timeseries'].columns.values:
                    if col.split('..')[0] == sk['label']:
                        outflow_args['variable_costs'] = nd['timeseries'][col]
            else:
                outflow_args['variable_costs'] = sk['variable_costs']

            if sk['emission_series']:
                for col in nd['timeseries'].columns.values:
                    if col.split('.')[0] == sk['label']:
                        outflow_args['emission_factor'] = nd['timeseries'][col]
            else:
                outflow_args['emission_factor'] = \
                    np.full(nd['general']['timesteps'][0], sk['emissions'])

            nodes.append(
                solph.Sink(
                    label=sk['label'], inputs={busd[sk['from']]: solph.Flow(
                        **outflow_args)})
            )

    # Create Transformer objects from 'transformers' table
    for i, t in nd['transformer'].iterrows():
        if t['active']:

            # Transformer with 1 Input and 1 Output
            if t['in_2'] == 0 and t['out_2'] == 0:

                if t['invest']:

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    # calculation epc
                    epc_t = economics.annuity(
                        capex=t['capex'], n=t['n'],
                        wacc=nd['general']['interest rate'][0]) * nd[
                            'general']['timesteps'][0] / 8760

                    # create
                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'],
                                summed_max=t['in_1_sum_max'],
                                max=t['max'],
                                investment=solph.Investment(
                                    ep_costs=epc_t + t['service']*(
                                            nd['general'][
                                                'timesteps'][0]/8760),
                                    maximum=t['max_invest'],
                                    minimum=t['min_invest']))},
                            conversion_factors={
                                busd[t['out_1']]: t['eff_out_1']})
                    )

                else:
                    # create
                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                nominal_value=t['installed'],
                                max=t['max'],
                                summed_max=t['in_1_sum_max'],
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'])},
                            conversion_factors={
                                busd[t['out_1']]: t['eff_out_1']})
                    )

            # Transformer with 1 Input and 2 Output
            if t['in_2'] == 0 and t['out_2'] != 0:

                if t['invest']:
                    # calculation epc
                    epc_t = economics.annuity(
                        capex=t['capex'], n=t['n'],
                        wacc=nd['general']['interest rate'][0]) *\
                          nd['general']['timesteps'][0] / 8760

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    # create
                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                summed_max=t['in_1_sum_max'],
                                max=t['max'],
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'],
                                investment=solph.Investment(
                                    ep_costs=epc_t + t['service']*(nd[
                                        'general']['timesteps'][0] / 8760),
                                    maximum=t['max_invest'],
                                    minimum=t['min_invest'])),
                                busd[t['out_2']]: solph.Flow()
                            },
                            conversion_factors={
                                busd[t['out_1']]: t['eff_out_1'],
                                busd[t['out_2']]: t['eff_out_2']
                            })
                    )

                else:
                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow()},
                            outputs={
                                busd[t['out_1']]: solph.Flow(
                                    summed_max=t['in_1_sum_max'],
                                    variable_costs=t['variable costs'],
                                    emissions=t['emissions'],
                                    max=t['max'],
                                    nominal_value=t['installed']),
                                busd[t['out_2']]: solph.Flow()
                            },
                            conversion_factors={
                                busd[t['out_1']]: t['eff_out_1'],
                                busd[t['out_2']]: t['eff_out_2']
                            })
                    )

            # Transformer with 2 Input and 1 Output
            if t['in_2'] != 0 and t['out_2'] == 0:

                if t['invest']:
                    # calculation epc
                    epc_t = economics.annuity(
                        capex=t['capex'], n=t['n'],
                        wacc=nd['general']['interest rate'][0]) *\
                          nd['general']['timesteps'][0] / 8760

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    # create
                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow(),
                                    busd[t['in_2']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                summed_max=t['in_1_sum_max'],
                                max=t['max'],
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'],
                                investment=solph.Investment(ep_costs=epc_t+t[
                                    'service']*(nd[
                                        'general']['timesteps'][0] / 8760),
                                    maximum=t['max_invest'],
                                    minimum=t['min_invest']))},
                            conversion_factors={
                                busd[t['in_1']]: t['eff_in_1'],
                                busd[t['in_2']]: t['eff_in_2'],
                                busd[t['out_1']]: t['eff_out_1']
                            })
                    )

                else:

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow(),
                                    busd[t['in_2']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                summed_max=t['in_1_sum_max'],
                                max=t['max'],
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'],
                                nominal_value=t['installed'])},
                            conversion_factors={
                                busd[t['in_1']]: t['eff_in_1'],
                                busd[t['in_2']]: t['eff_in_2'],
                                busd[t['out_1']]: t['eff_out_1']
                            })
                    )

            # Transformer with 2 Input and 2 Output
            if t['in_2'] != 0 and t['out_2'] != 0:

                if t['invest']:
                    # calculation epc
                    epc_t = economics.annuity(
                        capex=t['capex'], n=t['n'],
                        wacc=nd['general']['interest rate'][0]) *\
                          nd['general']['timesteps'][0] / 8760

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    # create
                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow(),
                                    busd[t['in_2']]: solph.Flow()},
                            outputs={busd[t['out_1']]: solph.Flow(
                                summed_max=t['in_1_sum_max'],
                                max=t['max'],
                                variable_costs=t['variable costs'],
                                emissions=t['emissions'],
                                investment=solph.Investment(ep_costs=epc_t + t[
                                    'service']*(nd[
                                     'general']['timesteps'][0] / 8760),
                                    maximum=t['max_invest'],
                                    minimum=t['min_invest'])
                                ),
                                busd[t['out_2']]: solph.Flow()},
                            conversion_factors={
                                busd[t['in_1']]: t['eff_in_1'],
                                busd[t['in_2']]: t['eff_in_2'],
                                busd[t['out_1']]: t['eff_out_1'],
                                busd[t['out_2']]: t['eff_out_2']
                            })
                    )

                else:

                    if t['max'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('..')[0] == t['label']:
                                t[col.split('..')[1]] = nd['timeseries'][col]

                    if t['eff_out_1'] == 'series':
                        for col in nd['timeseries'].columns.values:
                            if col.split('.')[0] == t['label']:
                                t[col.split('.')[1]] = nd['timeseries'][col]

                    nodes.append(
                        solph.Transformer(
                            label=t['label'],
                            inputs={busd[t['in_1']]: solph.Flow(),
                                    busd[t['in_2']]: solph.Flow()},
                            outputs={
                                busd[t['out_1']]: solph.Flow(
                                    summed_max=t['in_1_sum_max'],
                                    max=t['max'],
                                    variable_costs=t['variable costs'],
                                    emissions=t['emissions'],
                                    nominal_value=t['installed']),
                                busd[t['out_2']]: solph.Flow()},
                            conversion_factors={
                                busd[t['in_1']]: t['eff_in_1'],
                                busd[t['in_2']]: t['eff_in_2'],
                                busd[t['out_1']]: t['eff_out_1'],
                                busd[t['out_2']]: t['eff_out_2']
                            })
                    )

    # create storages
    for i, s in nd['storages'].iterrows():
        if s['active']:
            if s['invest']:
                # calculate epc
                epc_s = economics.annuity(
                    capex=s['capex'], n=s['n'],
                    wacc=nd['general']['interest rate'][0]) * \
                        nd['general']['timesteps'][0] / 8760

                # create Storages
                nodes.append(
                    solph.components.GenericStorage(
                        label=s['label'],
                        inputs={busd[s['bus']]: solph.Flow()},
                        outputs={busd[s['bus']]: solph.Flow()},
                        loss_rate=s['capacity_loss'],
                        invest_relation_input_capacity=s[
                            'invest_relation_input_capacity'],
                        invest_relation_output_capacity=s[
                            'invest_relation_output_capacity'],
                        inflow_conversion_factor=s['inflow_conversion_factor'],
                        outflow_conversion_factor=s[
                            'outflow_conversion_factor'],
                        investment=solph.Investment(ep_costs=epc_s,
                                                    maximum=s['max_invest'])))
            else:
                # create Storages
                if type(s['initial_storage_level']) is str or 'nan':
                    isl = None
                else:
                    isl = s['initial_storage_level']

                nodes.append(
                    solph.components.GenericStorage(
                        label=s['label'],
                        inputs={busd[s['bus']]: solph.Flow()},
                        outputs={busd[s['bus']]: solph.Flow()},
                        loss_rate=s['capacity_loss'],
                        nominal_storage_capacity=s['capacity'],
                        initial_storage_level=isl,
                        inflow_conversion_factor=s['inflow_conversion_factor'],
                        outflow_conversion_factor=s[
                            'outflow_conversion_factor'],
                        ))

    return nodes


def setup_es(excel_nodes=None):
    # Initialise the Energy System
    logger.define_logging()
    logging.info('Initialize the energy system')

    number_timesteps = excel_nodes['general']['timesteps'][0]

    start = pd.to_datetime('1/1/2018')

    date_time_index = pd.date_range(start,
                                    periods=number_timesteps,
                                    freq='H')
    energysystem = solph.EnergySystem(timeindex=date_time_index)

    logging.info('Create oemof objects')

    # create nodes from Excel sheet data with create_nodes function
    my_nodes = create_nodes(nd=excel_nodes)

    # add nodes and flows to energy system
    energysystem.add(*my_nodes)

    return energysystem
