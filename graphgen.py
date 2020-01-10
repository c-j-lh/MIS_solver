from os.path import isfile

from tqdm import *
import igraph

from utils.graph import *


shapes = []
with open('tmp_scripts/frb_nm.txt', 'r') as file:
    for line in file:
        shapes.append([int(token) for token in line.split()])
        

list_ = [(100, 200, 1362.5, 2525, 3688, 4850), (200, 400, 5225, 10050, 14875, 19700), (400, 800, 20450, 40100, 59750, 79400), (800, 1600, 80900, 160200, 239500, 318800), (1600, 3200, 321800, 640400, 959000, 1277600)]

#for i, (n, m) in enumerate(tqdm(shapes)):
#    igraph_ = igraph.Graph.Erdos_Renyi(n=n, m=m)
#    g = from_igraph(igraph_)
#    write_graph(g, 'data/random/frb{:02d}'.format(i)) 


for graph_no in range(10):
    igraph_ = igraph.Graph.Erdos_Renyi(n=300, m=750)
    g = from_igraph(igraph_)
    write_graph(g, 'data/random/300_750_{}'.format(graph_no))
    
#for n in tqdm([100, 200, 400, 800, 1600],unit='n'):
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#        for graphname in tqdm(os.listdir('data/frb')):
#            print(graphname, len(graphname))
#            if len(graphname) == len('frb30-15-0.mis'):
#                graphname = 'data/frb/' + graphname
#                graphs.append(read_graph(graphname))
#                graphnames.append(filename)
#        for i in trange(40):
#            graphname = 'data/random/frb{:02d}'.format(i)
#            graphs.append(read_graph(graphname))
#            graphnames.append(graphname)
#            
#    for mp in tqdm([int(i*n) for i in (10, 2.5)], unit='mp', leave=False):
#        for i in trange(10, unit='instance', leave=False):
#            if isfile('data/random/{}_{}_{}'.format(n,mp,i)): continue
#            if isinstance(mp, float):
#                igraph_ = igraph.Graph.Erdos_Renyi(n=n, p=mp)
#            else:
#                igraph_ = igraph.Graph.Erdos_Renyi(n=n, m=mp)
#            g = from_igraph(igraph_)
#            write_graph(g, 'data/random/{}_{}_{}'.format(n,mp,i))
