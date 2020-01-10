#!/usr/bin/env python3
from argparse import ArgumentParser
import time as time_

import torch
import torch.multiprocessing
from torch.multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)
import numpy as np
from tqdm import tqdm, trange

from config import device, use_dense
from utils.graph import *
from utils.timer import Timer
from mcts.mcts import MCTS
from mcts.mcts_node import MCTSNode
from gin.gin import GIN3
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse

#rollout
def rollout(mcts, mcts2, root_node, root_node2, stop_at_leaf=False):
    node = root_node
    node2 = root_node2
    v = -1
    finish = False
    count = 0
    while not finish:
        if node.is_end(): break
        children = node.raw_children()
        children2 = node2.raw_children()
        #child = np.argmax(children)
        #child2 = np.argmax(children2)
        vs = (children + children2)/2  # old: node.best_child()
        v = np.argmax(vs)
        count += 1
        #if child!=child2: print(children, child, children2, child2,vs,v,sep='\n')
        if node.children[v] is None:
            env = MISEnv() if use_dense else MISEnv_Sparse()
            env.set_graph(node.graph)
            next_graph, r, done, info = env.step(v)
            node.children[v] = MCTSNode(next_graph, mcts, idx=v, parent=node)
            node2.children[v] = MCTSNode(next_graph, mcts2, idx=v, parent=node2)
            if stop_at_leaf:
                finish = True
        node = node.children[v]
        node2 = node2.children[v]

    # backpropagate V: should I?
    '''
    V = node.state_value()
    V2 = node.state_value()
    while node is not root_node:
        V += 1
        V2 += 1
        mcts.update_parent(node, V)
        mcts2.update_parent(node, V2)
        node = node.parent
        node2 = node2.parent
    mcts.root_max = max(mcts.root_max, V)
    mcts2.root_max = max(mcts2.root_max, V2)
    '''
    return count

#graphnames = ["data/frb/frb30-15-{}.mis".format(i) for i in range(5)] \
#             + ["data/random/{}_{}_{}".format(n, m,i) for n in (100,200,400,800,1600) for m in (0.2,0.5,0.8) for i in range(3)]
#graphs = list(map(read_graph, graphnames))

gnn = GIN3(layer_num=6)
gnn.load_state_dict(torch.load('model/m200_0.pth'))
gnn.to(device)
gnn.eval()
mcts = MCTS(gnn)

gnn2 = GIN3(layer_num=6)
gnn2.load_state_dict(torch.load('model/n200_1.pth'))
gnn2.to(device)
gnn2.eval()
mcts2 = MCTS(gnn2)

time_limit = 600

"""
for graph_no, graph in enumerate(tqdm(graphs, unit='graph')):
    root_node = MCTSNode(graph, mcts)
    root_node2 = MCTSNode(graph, mcts2)
    results = []
    now = time_.process_time()
    count = 0
    while and count: #iter_num
        r = rollout(mcts, mcts2, root_node, root_node2)
        results.append(r)
        connt += 1
    graph_filename = '500_1250_{}'.format(graph_no)
    out = "{} {} iter={}\n".format(graph_filename, max(results), iter_num)
    with open('results/m200_0-n200_1', 'a') as file:
        file.write(out)
"""

def use_model(graph):
    root_node = MCTSNode(graph, mcts)
    root_node2 = MCTSNode(graph, mcts2)
    results = []
    now = time_.process_time()
    count = 0
    while time_.process_time()-now < time_limit or count==0: #iter_num
        r = rollout(mcts, mcts2, root_node, root_node2)
        results.append(r)
        count += 1
    return results, time_.process_time() - now


import pickle
if __name__ == "__main__":
    print('starting loading graphs')
    with open('graphs.pickle', 'rb') as file:
        graphs = pickle.load(file)
    graphnames = ["data/frb/frb30-15-{}.mis".format(i) for i in range(5)] \
                 + ["data/random/{}_{}_{}".format(n, m,i) for n in (100,200,400,800,1600) for m in (0.2,0.5,0.8) for i in range(3)]
    #graphs = list(map(read_graph, graphnames))
    #with open('graphs.pickle', 'wb') as file:
    #    pickle.dump(graphs, file)
    print('ending loading graphs')
    for graph, graphname in zip(tqdm(graphs, unit="graph", leave=False), graphnames):
        graph = graph.adj
        results, time = use_model(graph)
        out = "{} {} models={} {}={}\n".format(
            graphname, max(results),
            1, 'new_time', time_limit)
            
        with open('results/m200_0-n200_1','a') as f:
            f.write(out)
            print('\tgnn={} time={:.3f} iter_num={}'.format(0, time, len(results)), *results, file=f)
