"""
oemof application for research project quarree100.

Some parts of the code are adapted from
https://github.com/oemof/oemof-examples
-> excel_reader

see also: https://github.com/quarree100/q100_oemof_app

Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""

import pandas as pd
import oemof.outputlib as outputlib
import oemof.solph as solph
from matplotlib import pyplot as plt


def plot_buses(res=None, es=None):

    l_buses = []

    for n in es.nodes:
        type_name =\
            str(type(n)).replace("<class 'oemof.solph.", "").replace("'>", "")
        if type_name == "network.Bus":
            l_buses.append(n.label)

    for n in l_buses:
        bus_sequences = outputlib.views.node(res, n)["sequences"]
        bus_sequences.plot(kind='line', drawstyle="steps-mid", subplots=False,
                           sharey=True)
        plt.show()


def plot_storages_soc(res=None):

    nodes = [x for x in res.keys() if x[1] is None]
    node_storage_invest_label = [x[0].label for x in nodes if isinstance(
        x[0], solph.components.GenericStorage)
                           if hasattr(res[x]['scalars'], 'invest')]

    for n in node_storage_invest_label:
        soc_sequences = outputlib.views.node(res, n)["sequences"]
        soc_sequences = soc_sequences.drop(soc_sequences.columns[[0, 2]], 1)
        soc_sequences.plot(kind='line', drawstyle="steps-mid", subplots=False,
                           sharey=True)
        plt.show()


def plot_invest(res=None):

    # Transformer
    flows = [x for x in res.keys() if x[1] is not None]
    flows_invest = [x for x in flows if isinstance(
        x[0], solph.Transformer) if hasattr(
        res[x]['scalars'], 'invest')]

    df_trafo_invest = pd.Series(
        index=[x[0].label for x in flows_invest],
        data=[res[x]['scalars']['invest'] for x in flows_invest])

    df_trafo_invest.plot(kind='bar')
    plt.ylabel('Installed Capacity [kW]')
    plt.show()

    # Storages
    nodes = [x for x in res.keys() if x[1] is None]
    node_storage_invest = [x for x in nodes if isinstance(
        x[0], solph.components.GenericStorage)
                           if hasattr(res[x]['scalars'], 'invest')]
    df_storage_invest = pd.Series(
        index=[x[0].label for x in node_storage_invest],
        data=[res[x]['scalars']['invest'] for x in node_storage_invest])
    df_storage_invest.plot(kind='bar')
    plt.ylabel('Installed Capacity Storage [kWh]')
    plt.show()
