#!/usr/bin/env python3
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import time as time_
import pickle
from os import getpid
from os.path import isfile

import torch.multiprocessing
from torch.multiprocessing import Pool, set_start_method
import numpy as np
import torch
from tqdm import tqdm, trange

from config import device
from utils.graph import read_graph
import pickle_graphs
from mcts.mcts import MCTS
from gin.gin import GIN3

parser = ArgumentParser(description='test MIS gnns', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str)
parser.add_argument('--time', type=int, default=200)
parser.add_argument('--iter_num', type=int, default=-1)
parser.add_argument('--gnn_idxs', type=int, default=[0], nargs='+')
parser.add_argument('--graph_idxs', type=int, default=[0,1,2,3,4,5,6,7,8,9], nargs='+')
parser.add_argument('--multithreading', type=int, default=False)
parser.add_argument('-c', type=str, nargs='*')
# N_iter=1_{}.pth
# modified_p5_{}th.pth deprecated
# N_iter=dynamic_{}_100epoch.pth
# original_p5_{}th.pth
args = parser.parse_args()
model = args.model
graph_idxs = args.graph_idxs
time_limit = args.time
iter_num = args.iter_num
gnn_idxs = args.gnn_idxs
assert '.pth' not in model and '{}' not in model


def best_gin(idx):
    gnn = GIN3(layer_num=6, feature=8)
    gnn.load_state_dict(torch.load('model/'+model.format(idx)))
    gnn.to(device)
    gnn.eval()
    return gnn


def best_gins(gnn_idxs=range(10)):
    return [best_gin(idx) for idx in gnn_idxs] #10


def use_model(t):
    gnn, graph = t
    mcts = MCTS(gnn) #, performance=True

    start = time_.process_time()
    result = mcts.search(graph, 10) # original
    return result, time_.process_time() - start


if __name__ == "__main__":

    graphs, graphnames = pickle_graphs.load('plot-11_graphs.pickle')
    gnns = []
    for i in range(1):
    #for i in range(4, 200, 5):
        filename = 'model/' + model + '.pth'
        if not isfile(filename):
            break
        gnn = GIN3(layer_num=6, feature=8)
        try:
            gnn.load_state_dict(torch.load(filename))
            gnn.to(device)
            gnn.eval()
            gnns.append(gnn)
            print(filename, 'worked')
        except Exception as e:
            print(e.args[0])
            print(filename, 'failed')
    print(len(gnns), 'epochs to \'plot\'\nWaiting 4s before continuing...\n\n\n')
    time_.sleep(4)
    print('continuing:\n')

    all_results = []
    for epoch, gnn in enumerate(tqdm(gnns, unit='epoch')):
        results = []
        for graph, graphname in zip(tqdm(graphs, unit="graph", leave=False), graphnames):
            graph = graph.adj
            result, time = use_model((gnn, graph))
            results.append(result)
             
        all_results.append(results)
        with open('log/' + model + '.plot-11.pickle', 'wb') as f:
            pickle.dump(all_results, f)
