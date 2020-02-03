"""
oemof.solph application for an emission constraint investment- and unit
commitment energy system optimization at urban scale.
The investment options and all data can be found in the
.xlsx data file (data/model_data.xlsx).

Johannes Röder <johannes.roeder@uni-bremen.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""

import logging
from matplotlib import pyplot as plt
import oemof.solph as solph
import oemof.outputlib as outputlib
import modules.setup_solve_model
import modules.postprocessing

# getting path to model data
path_to_data = 'data/'

# selecting input scenario file
filename = path_to_data + 'model_data.xlsx'

# reading data from excel file with data read function
node_data = modules.setup_solve_model.nodes_from_excel(filename)

# setting up energy system
e_sys = modules.setup_solve_model.setup_es(excel_nodes=node_data)

# Optimise the energy system
logging.info('Optimise the energy system')

# initialise the operational model
om = solph.Model(e_sys)

# Global CONSTRAINTS: emission limit
solph.constraints.generic_integral_limit(
    om, keyword='emission_factor', limit=505000)

logging.info('Solve the optimization problem')
om.solve(solver='cbc', solve_kwargs={'tee': True})

# plot the Energy System
try:
    import pygraphviz
    from modules import graph_model as gm
    from oemof.graph import create_nx_graph
    import networkx as nx
    grph = create_nx_graph(e_sys)
    pos = nx.drawing.nx_agraph.graphviz_layout(grph, prog='neato')
    gm.plot_graph(pos, grph)
    plt.show()
    logging.info('Energy system Graph OK')
except ImportError:
    logging.info('Module pygraphviz not found: Graph was not plotted.')

logging.info('Store the energy system with the results.')
# add results to the energy system to make it possible to store them.
e_sys.results['main'] = outputlib.processing.results(om)
e_sys.results['meta'] = outputlib.processing.meta_results(om)

# store energy system with results
# e_sys.dump(dpath=path_to_results, filename='results_xy')
total_emission = om.integral_limit_emission_factor()
print('Total Emission [kg]')
print(total_emission)

results = e_sys.results['main']

# plot investment results of flows and storages
modules.postprocessing.plot_invest(res=results)

# plot the in- and outflow of all buses
modules.postprocessing.plot_buses(res=results, es=e_sys)

# plot storage SoC
modules.postprocessing.plot_storages_soc(res=results)

# calculate relative emission and cost values
total_emission = om.integral_limit_emission_factor()
print('Total Emission [kg]')
print(total_emission)

e_total = node_data['timeseries']['demand_elec.actual_value'].sum() + \
    node_data['timeseries']['demand_heat.actual_value'].sum()

total_costs = e_sys.results['meta']['objective']

print('')
print('Relative Emission [kg/kWh]')
print(total_emission / e_total)
print('')
print('Relative Costs [€/kWh]')
print(total_costs / e_total)
