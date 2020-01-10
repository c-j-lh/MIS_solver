import sys
import os
from time import sleep

from igraph import Graph
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.graph import read_graph

values, degrees = [], []
centralities = []
for i in range(10):
    g = read_graph('../data/random/100_250_{}'.format(i))
    #degrees.append([len(neighbors) for neighbors in g.tmp])
    graph = Graph()
    graph.add_vertices(g.n)
    graph.add_edges([(n, neighbor) for n in range(g.n) for neighbor in g.tmp[n] if n < neighbor])
    centralities.append(graph.evcent())
    degrees.append([len(neighbors) for neighbors in g.tmp])

with open('nodewit3.txt','r') as file:
    for line in file.readlines():
        if line=='' or line=='\n': values.append([])
        else:
            values[-1].extend([float(token) for token in line.split(' ') if token])

print(len(centralities), len(degrees))
for i, (type_, data) in enumerate(zip(('centrality', 'degree'), (centralities, degrees))):
    print(i)
    plt.figure(i)
    for figure in range(10):
        print(len(data))
        #plt.ion()
        plt.plot(data[figure], values[figure], 'o')
        plt.xlabel('{} per node'.format(type_.title()))
        plt.ylabel('Model-predicted values, $P$, per node')
        plt.title('Graph of model-predicated values,\n$P$, against {}, per node\nfor 100-nodes-250-edges graphs'.format(type_))
    plt.tight_layout()
    plt.savefig('../images/{}_correlation.png'.format(type_))
    plt.savefig('../images/{}_correlation.svg'.format(type_))
    #plt.clf()
plt.show()

        
#values = list(zip(*values))
#degrees = list(zip(*degrees))
#with open('values.csv','w') as file:
#    for hundredth in values:
#        file.write(','.join(map(str,hundredth))+'\n')
#with open('degrees.csv','w') as file:
#    for hundredth in degrees:
#        file.write(','.join(map(str,hundredth))+'\n')
