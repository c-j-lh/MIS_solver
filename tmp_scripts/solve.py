from igraph import Graph
from tqdm import *

with open('MIS.txt','a') as outfile:
    for m in tqdm((100, 300, 500, 700, 900),unit='type'):
        for i in trange(10, unit='graph'):
            with open('data/random/50_{}_{}'.format(m,i), 'r') as file:
                n,e = map(int, file.readline().split())
                graph = Graph()
                graph.add_vertices(n)
                for j in range(e):
                    a,b = map(int, file.readline().split())
                    graph.add_edges([(a,b)])

            sets = graph.largest_independent_vertex_sets()
            size = len(sets[0])
            print('data/random/50_{}_{}: {}'.format(m,i,size), file=outfile)
            cnt = 0
            for _set in sets:
                print(_set, file=outfile)
                cnt += 1
                if cnt==10: break
            print(file=outfile)
