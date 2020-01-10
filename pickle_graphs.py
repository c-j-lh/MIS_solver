import pickle
import os

from tqdm import tqdm, trange

from utils.graph import read_graph


def save():
    filename = '24-2.5-10_{}.pickle'
    filname = 'data/pickled/' + filename
    with open(filename.format('graphs'), 'wb') as file, open(filename.format('graphfilenames'), 'wb') as filenamefile:
        graphs = []
        graphnames = []
        #for graphname in tqdm(sorted(os.listdir('data/frb'))):
        #    if len(graphname) == len('frb30-15-0') and graphname[9] == '0' and int(graphname[3:5]) % 2:
        #        graphname = 'data/frb/' + graphname
        #        graphs.append(read_graph(graphname))
        #        graphnames.append(graphname)

        #for i in trange(5, 40, 10):
        #    graphname = 'data/random/frb{:02d}'.format(i)
        #    graphs.append(read_graph(graphname))
        #    graphnames.append(graphname)
            
        for n in tqdm((200, 400), unit='size'):
            for m in (int(2.5*n), 10*n):
                graph_no = 0
                graphname = 'data/random/{}_{}_{}'.format(n, m, graph_no)
                graphs.append(read_graph(graphname))
                graphnames.append(graphname)

        print(len(graphs), 'graphs pickled')
        pickle.dump(graphs, file)
        print('dumping to ', file.name)
        pickle.dump(graphnames, filenamefile)

def load(filename):
    """Returns graphs, graphnames"""
    filename = 'data/pickled/' + filename
    with open(filename.format('graphs'), 'rb') as file, open(filename.format('graphfilenames'), 'rb') as filenamefile:
        return pickle.load(file), pickle.load(filenamefile)

if __name__ == '__main__':
    save()
