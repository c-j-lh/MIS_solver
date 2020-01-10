import pickle
import os
from itertools import zip_longest

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.ion()
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]
linestyle_tuple = dict(linestyle_tuple)
linestyles = ['--', '-', '-.', 'densely dashdotdotted', 'dashdotdotted', 'loosely dashdotdotted',
              'densely dotted', 'loosely dotted']
linestyles = [ls if ls not in linestyle_tuple else linestyle_tuple[ls] for ls in linestyles]

# 'cinc', 'train100', 'train200'
# 'hardened', 'chardened'
# 'dynUCB', 'chardened', 'hardened'
# 'cinc', 'train100', 'train200', 'dynUCB', 'chardened', 'hardened'
setup_names = ('novacinc', 'novatrain100', 'novatrain200', 'novatrain300', 'hardened', 'hardened4')  #, 'chardened', 'dynUCB') 
lay_names = ('Curriculum-Increasing', 'Train100', 'Train200', 'Train300', 'Curriculum-Hard-Graphs', 'Hard-Graphs', 'Dynamic-UCB-Weights')
with open('data/pickled/plot-7_graphfilenames.pickle', 'rb') as file:
    graphnames = pickle.load(file)

ideal = [[44, 45, 43, 45, 44], # 100_250_0 - 100_250_4
         [45, 53, 35, 59, 44, None, None],  # plot-11
         [220, 221, 220, 222, 214],  # 500_1250_0 - 500_1250_4
         [428, 422, 434, 437, 434]  # 1000_2500_0 - 1000_2500_4 (not necc. best values)
        ][1]
if ideal:
    if None not in ideal:
        ideal.append(sum(ideal) / len(ideal))

# Get train timings
times = []
for setup_name in setup_names:
    with open('ctrain_time/{}_0.pickle'.format(setup_name), 'rb') as file:
        times.append([i/3600 for i in pickle.load(file)])
        times[-1] = times[-1][0::5]

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

for i, setup in enumerate(raw_logs):
    np.transpose(setup, (2, 1, 0))
by_graph = [np.transpose(setup, (2, 1, 0)) for setup in raw_logs] # now 5 graphs, 100-400 epochs, 1-10 models
by_graph = [setup.mean(2) for setup in by_graph] # now 4-8 graphs, 200 epochs

average  = [np.transpose(setup, (1, 2, 0)) for setup in raw_logs] # now 100-400 epochs, 4-8 graphs, 1-10 models
average = [setup.reshape(setup.shape[0], -1).mean(1) for setup in average]

# Plotting average graphs
plt.figure(no_graphs)
plot_epochns = []
for i, (setup_name, setup, raw_log, time, linestyle, lay_name) in enumerate(zip(setup_names, average, raw_logs, times, linestyles, lay_names)):
    if len(time) == len(setup)-1:
        times[i] = time = [0.0] + time
        plot_epochns.append(range(len(time)))
    else:
        plot_epochns.append(range(1, len(time)+1))
    if len(time) != len(setup):
        print(setup_name, len(time), len(setup))
        times[i] = time = [2000] * len(setup)
    plt.plot(time, setup, label=lay_name, linestyle=linestyle)


# Plotting results for each graph
max_epoch = max(raw_log.shape[1] for raw_log in raw_logs)
for setup_name, setup, raw_log, time, linestyle, lay_name in zip(setup_names, by_graph, raw_logs, times, linestyles, lay_names):
    for graph_no, graph  in enumerate(setup):
        label = '{}: {} models, {} epochs'.format(setup_name, raw_log.shape[0], raw_log.shape[1])
        plt.figure(graph_no)
        plt.plot(time, graph, label=lay_name, linestyle=linestyle)

# Plotting average and labels
for graph_no, graphname in zip_longest(range(no_graphs + 1), graphnames):
    plt.figure(graph_no)
    if ideal and graph_no < len(ideal):
        #plt.plot([0, 5*max_epoch - 1], 2 * [ideal[graph_no]], label="Actual $\\alpha(G)$")
        plt.plot([0, max(max(time) for time in times)], 2 * [ideal[graph_no]], label=r"Actual $\alpha(G)$")
    plt.xlabel('Epochs trained')
    plt.ylabel(r'Average $\alpha(G)$ predicted')  # average MIS size predicted by setup_names
    plt.title('Graph of setups\' performance during training against epochs\n{} during training'
              .format('on graph, G: ' + graphname if graphname else 'averaged across graphs'))
    plt.legend()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    fstr = 'images/new-by-epoch_{}'
    #plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.png', dpi=1500)
    #plt.savefig(fstr.format(graph_no if graph_no != no_graphs else 'average') + '.svg')
