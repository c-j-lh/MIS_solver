import pickle
import os
from itertools import zip_longest, product

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

#plt.ion()
linestyles = ['-', ':', '--']
#linewidths = [0.5, 2.0, 3.5, 5.0]
linewidths = [2.0, 3.5, 5.0, 6.0]
options = [{'linewidth': linewidth, 'linestyle': linestyle}
           for linewidth, linestyle in product(linewidths, linestyles)]

counts =  [0] * 1000
def fplot(*args, **kwargs):
    "friendly plot"
    count = counts[plt.gcf().number]
    plt.plot(*args, **kwargs, **(options[count]))
    print(kwargs.values(), plt.gcf().number, count)
    
    counts[plt.gcf().number] += 1

# 'cinc', 'train100', 'train200'
# 'hardened', 'chardened'
# 'dynUCB', 'chardened', 'hardened'
# 'cinc', 'train100', 'train200', 'dynUCB', 'chardened', 'hardened'
setup_names = ('c6', 'c5', 'c3', 'c4', 'c7', 'c8', 'q11')  #, 'chardened', 'dynUCB') 
lay_names = ('$Q:V = 10-e$', '$Q:V = 11-e$', '$Q:V = 12-e$', '$Q:V = 13-e$', '$Q:V = 10\\times0.9^e$', 'Q:V = $11\\times0.9^e$', '$Q:V = 11$', '$Q:V = 1$', '$Q:V = 10$', '$Q:V = 9$', '$Q:V = 5$')
graphnames = ['100_250_{}'.format(i) for i in range(5)]

ideal = [[44, 45, 43, 45, 44], # 100_250_0 - 100_250_4
         [45, 53, 35, 59, 44, None, None],  # plot-11
         [220, 221, 220, 222, 214],  # 500_1250_0 - 500_1250_4
         [428, 422, 434, 437, 434],  # 1000_2500_0 - 1000_2500_4 (not necc. best values)
         [None] * 4,
        ][4]
if ideal:
    if None not in ideal:
        ideal.append(sum(ideal) / len(ideal))

logs = []
raw_logs = []
for setup_name in setup_names:
    raw_logs.append([])
    setup_name += '_{}.pickle'
    if setup_name[:4]=='n200':
        raw_logs[-1].append(pickle.load(open('log/'+setup_name,'rb')))
        continue

    for model_no in range(1):
        try:
            print('log/' + setup_name.format(model_no), 'rb')
            with open('log/' + setup_name.format(model_no), 'rb') as file:
                raw_logs[-1].append(pickle.load(file))
        except FileNotFoundError:
            pass
    if not raw_logs[-1]:
        raise NameError('Model not found: {}'.format(setup_name))

raw_logs = [np.array([[[sum(graph)/len(graph) for graph in epoch] for epoch in model] for model in setup]) for setup in raw_logs]
no_graphs = raw_logs[0].shape[-1]
print(no_graphs, 'graphs')
# ? setup_names, (1-10 models, 100-400 epochs, 4-8 graphs)

by_graph = [np.transpose(setup, (2, 1, 0)) for setup in raw_logs] # now 5 graphs, 100-400 epochs, 1-10 models
by_graph = [setup.mean(2) for setup in by_graph] # now 4-8 graphs, 200 epochs

average  = [np.transpose(setup, (1, 2, 0)) for setup in raw_logs] # now 100-400 epochs, 4-8 graphs, 1-10 models
average = [setup.reshape(setup.shape[0], -1).mean(1) for setup in average]

# Plotting average graphs
plt.figure(no_graphs)
plot_epochns = []
for i, (setup_name, setup, raw_log, lay_name) in enumerate(zip(setup_names, average, raw_logs, lay_names)):
    #if len(time) == len(setup)-1:
    #    times[i] = time = [0.0] + time
    #    plot_epochns.append(range(len(time)))
    #else:
    #    plot_epochns.append(range(1, len(time)+1))
    #if len(time) != len(setup):
    #    print(setup_name, len(time), len(setup))
    #    times[i] = time = [2000] * len(setup)
    setup = setup[:71]
    fplot(setup, label=lay_name)


# Plotting results for each graph
max_epoch = max(raw_log.shape[1] for raw_log in raw_logs)
for setup_name, setup, raw_log, lay_name in zip(setup_names, by_graph, raw_logs, lay_names):
    for graph_no, graph  in enumerate(setup):
        label = '{}: {} models, {} epochs'.format(setup_name, raw_log.shape[0], raw_log.shape[1])
        plt.figure(graph_no)
        fplot(graph, label=lay_name)

# Plotting average and labels
for graph_no, graphname in zip_longest(range(no_graphs + 1), graphnames):
    plt.figure(graph_no)
    if ideal and graph_no < len(ideal):
        #fplot([0, 5*max_epoch - 1], 2 * [ideal[graph_no]], label="Actual $\\alpha(G)$")
        fplot([0, 70], 2 * [ideal[graph_no]], label=r"Actual $\alpha(G)$")
    plt.xlabel('Epochs trained')
    plt.ylabel(r'Average $\alpha(G)$ predicted')  # average MIS size predicted by setup_names
    #plt.title('Graph of setups\' performance during training against epochs\n{} during training'
    #          .format('on graph, G: ' + graphname if graphname else 'averaged across graphs'))
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    fstr = 'images/newUCB_{}'
    #plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.png', dpi=1500)
    plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.svg')

plt.show()
