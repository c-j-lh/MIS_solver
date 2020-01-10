from igraph import *
from tqdm import *
import matplotlib as plt
import time

with open('MIS.txt','a') as outfile:
    for n in range(10,100):
        #with open('data/random/10_25_{}'.format(i), 'r') as file:
        #    n,e = map(int, file.readline().split())
        #    graph = Graph()
        #    graph.add_vertices(n)
        #    for j in range(e):
        #        a,b = map(int, file.readline().split())
        #        graph.add_edges([(a,b)])
        time1 = time.time()
        graph = Graph.Erdos_Renyi(n=n, m=int(2.5*n))
        graph.largest_independent_vertex_sets()
        print(n, time.time()-time1)

        #print('data/random/10_25/{}'.format(i), file=outfile)
        #for _set in graph.largest_independent_vertex_sets():
        #    print(_set, file=outfile)
        #print(file=outfile)
