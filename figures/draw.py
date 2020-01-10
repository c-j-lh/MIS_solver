import networkx as nx
import matplotlib.pyplot as plt
from igraph import Graph
import random #import numpy as np


'''G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

val_map = {'A': 1.0,
           'D': 0.5714285714285714,
           'H': 0.0}'''

n, p = 8, 0.5

plt.ion()
'''while True:
    try:'''
mIgraph = Graph.Erdos_Renyi(n=n, m=8)
mis = mIgraph.largest_independent_vertex_sets()[0]

mIgraph2 = mIgraph.copy()
edges = set((j, i) for i in range(mIgraph.vcount()) for j in range(i)) \
        .difference(mIgraph.get_edgelist())
edges = list(edges)
random.shuffle(edges)
mIgraph2.add_edges(edges[:6])


mIgraph3 = mIgraph2.copy()
mIgraph3.add_edges(edges[6:12])

pos = None
for i in range(2):
    for figure, mIgraph in enumerate([mIgraph, mIgraph2, mIgraph3]):
        mNgraph = nx.Graph()
        mNgraph.add_nodes_from(range(mIgraph.vcount()))
        mNgraph.add_edges_from(mIgraph.get_edgelist())
        values = [(0.2 if False else 0.0) if node in mis else 0.0 \
                  for node in range(n)]

        plt.figure(figure)
        if pos is None:
            pos = nx.circular_layout(mNgraph)
        nx.draw_networkx_nodes(mNgraph, pos, cmap=plt.get_cmap('jet'), 
                               node_color=values, node_size=500)
        nx.draw_networkx_edges(mNgraph, pos, edgelist=mIgraph.get_edgelist())
        fig = plt.gcf()
        fig = plt.gcf()
        #plt.savefig(f'{n}-{m}-{"un" if color else ""}colored.svg')
        #plt.savefig(f'{n}.png')
        plt.show()


'''plt.figure(1)
ps = [0.5, 0.4, 0.8, 0.6, 0.3, 0.7, 0.9, 0.2]
labels = dict((i, p) for i, p in enumerate(ps))
nx.draw_networkx_nodes(mNgraph, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size=500
                       )
nx.draw_networkx_edges(mNgraph, pos, edgelist=mIgraph.get_edgelist())
nx.draw_networkx_labels(mNgraph, pos, labels, font_color='w')
'''

'''break
    except ValueError:
        continue'''
