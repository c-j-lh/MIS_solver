#!/usr/bin/env python3
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import time as time_
from os import getpid

import torch.multiprocessing
from torch.multiprocessing import Pool, set_start_method
import numpy as np
import torch
from tqdm import tqdm, trange
from config import device
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3

parser = ArgumentParser(description='test MIS gnns', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default="original_p5_{}th.pth")
parser.add_argument('--time', type=int, default=-1)
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


def best_gin(idx):
    gnn = GIN3(layer_num=6, feature=8, idx=idx)
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
    gnns = best_gins(gnn_idxs)

    graphs, graphnames = pickle_graphs.load('p&m-frbs&cmp_{}.pickle')

    for graph, graphname in zip(tqdm(graphs, unit="graph", leave=False), graphnames):
        graph = graph.adj
        results, times = [], []
        for count,gnn in enumerate(tqdm(gnns, leave=False, unit="gnn")):
            start = time_.process_time()
            pair = use_model((gnn, graph))
            results.append(pair[0])
            times.append(pair[1])

        out = "{} {} models={} {}={}\n".format(
            graphname, max(max(gnn_result) for gnn_result in results),
            len(gnn_idxs), 'iter_num' if iter_num!=-1 else 'new_time',
            iter_num if iter_num!=-1 else time_limit)
            
        with open('results/'+model,'a') as f:
            f.write(out)
            for gnn, (gnn_result, time) in enumerate(zip(results, times)):
                print('\tgnn={} time={:.3f} iter_num={}'.format(gnn, time, len(gnn_result)), *gnn_result, file=f)
