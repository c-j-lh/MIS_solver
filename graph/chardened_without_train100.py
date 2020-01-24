import pickle
import os
from itertools import zip_longest, product

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

#plt.ion()
linestyles = ['-', ':', '--']
#linewidths = [2.0, 3.5, 5.0]
linewidths = [2.0, 2.75, 3.5]
options = [{'linewidth': linewidth, 'linestyle': linestyle}
           for linewidth, linestyle in product(linewidths, linestyles)]
counts = [0] * 1000
def fplot(*args, **kwargs):
    "friendly plot"
    count = counts[plt.gcf().number]
    plt.plot(*args, **kwargs, **(options[count]))
    counts[plt.gcf().number] += 1

# 'cinc', 'train100', 'train200'
# 'hardened', 'chardened'
# 'dynUCB', 'chardened', 'hardened'
# 'cinc', 'train100', 'train200', 'dynUCB', 'chardened', 'hardened'
setup_names = ('hardened', 'chardened')  #, 'chardened', 'dynUCB') 
lay_names = ('Hard-Graphs', 'Curriculum-Hard-Graphs') #, 'Hard-Graphs')
#with open('data/pickled/plot-11_graphfilenames.pickle', 'rb') as file:
#    graphnames = pickle.load(file)
graphnames = ['200_500_{}'.format(i) for i in range(4)]

ideal = [[44, 45, 43, 45, 44], # 100_250_0 - 100_250_4
         [45, 53, 35, 59, 44, None, None],  # plot_11
         [220, 221, 220, 222, 214],  # 500_1250_0 - 500_1250_4
         [428, 422, 434, 437, 434],  # 1000_2500_0 - 1000_2500_4 (not necc. best values)
         [None, None, None, None, None],  # 24-2.5-10
        ][4]
if ideal:
    if None not in ideal:
        ideal.append(sum(ideal) / len(ideal))

# Get train timings
#times = []
#for setup_name in setup_names:
#    with open('ctrain_time/{}_0.pickle'.format(setup_name), 'rb') as file:
#        times.append([i/3600 for i in pickle.load(file)])
#        times[-1] = times[-1][0::5]

logs = []
raw_logs = []
for setup_name in setup_names:
    raw_logs.append([])
    #if setup_name == 'dynUCB':
    #    setup_name += '_{}.merge.pickle'
    #else:
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

    if setup_name == 'dynUCB':
        raw_logs[-1] = raw_logs[-1][1:]
    if not raw_logs[-1]:
        raise NameError('Model not found: {}'.format(setup_name))

raw_logs = [np.array([[[sum(graph)/len(graph) for graph in epoch] for epoch in model] for model in setup]) for setup in raw_logs]
no_graphs = raw_logs[0].shape[-1]
print(no_graphs, 'graphs')
# ? setup_names, (1-10 models, 100-400 epochs, 4-8 graphs)

for i, setup in enumerate(raw_logs):
    np.transpose(setup, (2, 1, 0))
by_graph = [np.transpose(setup, (2, 1, 0)) for setup in raw_logs] # now 5 graphs, 100-400 epochs, 1-10 models
by_graph = [setup.mean(2) for setup in by_graph] # now 4-8 graphs, 200 epochs

average  = [np.transpose(setup, (1, 2, 0)) for setup in raw_logs] # now 100-400 epochs, 4-8 graphs, 1-10 models
average = [setup.reshape(setup.shape[0], -1).mean(1) for setup in average]

# Plotting average graphs
plt.figure(no_graphs)
plot_epochns = []
for i, (setup_name, setup, raw_log, lay_name) in enumerate(zip(setup_names, average, raw_logs, lay_names)):
    if setup_name == 'hardened':
        print(setup, lay_name, sep='\n\n')
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
    if ideal and graph_no < len(ideal) and ideal[graph_no]:
        #fplot([0, 5*max_epoch - 1], 2 * [ideal[graph_no]], label="Actual $\\alpha(G)$")
        fplot([0, max_epoch], 2 * [ideal[graph_no]], label=r"Actual $\alpha(G)$")
    plt.xlabel('Epochs trained')
    plt.ylabel(r'Average $\alpha(G)$ predicted')  # average MIS size predicted by setup_names
    #plt.title('Graph of setups\' performance during training against epochs\n{} during training'
    #          .format('on graph, G: ' + graphname if graphname else 'averaged across graphs'))
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    fstr = 'images/chardened_{}'
    plt.legend(prop={'size': 14})
    plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.svg')
    plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.png', dpi=600)
plt.show()
