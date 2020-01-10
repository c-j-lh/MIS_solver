#!/usr/bin/env python3
#dynamic
from config import device
import numpy as np
import torch
from utils.graph import read_graph, read_test_graphs, generate_random_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer import MCTSTrainer
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter
from tqdm import trange, tqdm
from argparse import ArgumentParser
from datetime import datetime
from os import listdir
import time as time_
import pickle
from contextlib import redirect_stderr

#with open('log/new_models.txt','a') as logfile:
#    logfile.write('\nStarting training: {}.\nTime: {}\nsetup_name: {} (new dynamic)\n'.format(, str(datetime.now())[:-7], setup_name))


epochs = 200
base = 0.98**(100/epochs)
time_limit = 600

tqdm.write('starting to load graphs')
graph = read_graph('data/frb/frb30-15-0.mis').adj
tqdm.write('finished loading graphs')

# load gnn
gnn = GIN3(layer_num=6)
gnn.load_state_dict(torch.load('model/m200_0.pth'))
gnn.to(device)
gnn.eval()
mcts = MCTS(gnn, dynamic=1)
#trainer = MCTSTrainer(gnn, [graph], 'dynamic')


#results = mcts.search(graph, 10) # only needs to be done once
#tqdm.write('graph',':', max(results))
def test():
    now = time_.process_time()
    results = mcts.search_for_exp(graph, time_limit) 
    time = time_.process_time() - now
    out = "{} models={} {}={}\n".format(
        max(results),
        1, 'new_time', time_limit)
        
    with open('results/m200_0-n200_1','a') as f:
        tqdm.write(out)
        tqdm.write('\tgnn={} time={:.3f} iter_num={} {}\n'.format(0, time, len(results), ' '.join(map(str, results))))

test()
for epoch in trange(0, 0+10, unit='epoch', leave=False):
    #trainer.train2(graph, 10*base**epoch, batch_size=10, iter_p=2)
    mcts.train(graph, 10*base**epoch, batch_size=10, iter_p=2, stop_at_leaf=True)
    tqdm.write('Epoch: {}'.format(epoch))
    test()


#for graph_no, graph in enumerate(tqdm(graphs, unit='graph', leave=False)):
#tqdm.write('graph',':', max(mcts.search(graph, 10)))
