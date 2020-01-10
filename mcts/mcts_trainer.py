import os
from os import listdir
from os.path import isfile
import pickle
import time as time_

from tqdm import tqdm
import numpy as np
import torch

from config import use_dense
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse
from mcts.mcts import MCTS
from utils.graph import read_graph
from utils.timer import Timer
from utils.gnnhash import GNNHash
from utils.nodehash import NodeHash

class MCTSTrainer:
    def __init__(self, gnn, test_graphs, filename, dynamic=-1, weights=(1, 0), new=False):
        # if it's a new model, check if it already exists
        if not isinstance(self, LoggingTrainer):
            raise Exception('deprecated')
        with open('log/new_models.txt','r') as logfile:
            saved = logfile.read()
        if new:
            if ('{}.pth'.format(filename) in listdir('model')) \
               or ('{}.pickle'.format(filename) in listdir('log')):
                raise Exception('model {} already exists'.format(filename))

        self.mcts = MCTS(gnn, dynamic=dynamic, weights=weights)
        if not test_graphs:
            with open('data/pickled/24-2.5-10_graphs.pickle', 'rb') as file:
                test_graphs = [graph.adj for graph in pickle.load(file)]
        self.test_graphs = test_graphs
        self.filename = filename
        self.test_result = []

        self.train_times = []
        self.ctrain_times = []
        self.train_graphs = []
        if isfile('test_result/{}.pickle'.format(self.filename)):
            with open('test_result/{}.pickle'.format(self.filename), 'rb') as file:
                self.test_result = pickle.load(file)
        if isfile('train_time/{}.pickle'.format(self.filename)):
            with open('train_time/{}.pickle'.format(self.filename), 'rb') as file:
                self.train_times = pickle.load(file)
        if isfile('ctrain_time/{}.pickle'.format(self.filename)):
            with open('ctrain_time/{}.pickle'.format(self.filename), 'rb') as file:
                self.ctrain_times = pickle.load(file)
        if isfile('train_graphs/{}.pickle'.format(self.filename)):
            with open('train_graphs/{}.pickle'.format(self.filename), 'rb') as file:
                self.train_graphs = pickle.load(file)
        if isfile('counts/{}.pickle'.format(self.filename)):
            with open('counts/{}.pickle'.format(self.filename), 'rb') as file:
                self.mcts.counts = pickle.load(file)

    def _train(self, graph, TAU, batch_size=110, iter_p=2, stop_at_leaf=True):
        start = time_.process_time()
        self.mcts.train(graph, TAU, batch_size=batch_size, iter_p=iter_p, stop_at_leaf=stop_at_leaf)

        self.train_times.append(time_.process_time() - start)
        os.makedirs('train_time', exist_ok=True)
        with open('train_time/{}.pickle'.format(self.filename), 'wb') as f:
            pickle.dump(self.train_times, f)

        self.ctrain_times.append(sum(self.train_times))
        os.makedirs('ctrain_time', exist_ok=True)
        with open('ctrain_time/{}.pickle'.format(self.filename), 'wb') as f:
            pickle.dump(self.ctrain_times, f)
        self._save_model()

        self.train_graphs.append(graph)
        os.makedirs('train_graphs', exist_ok=True)
        with open('train_graphs/{}.pickle'.format(self.filename), 'wb') as f:
            pickle.dump(self.train_graphs, f)

        os.makedirs('counts', exist_ok=True)
        with open('counts/{}.pickle'.format(self.filename), 'wb') as f:
            pickle.dump(self.mcts.counts, f)
        

    # rollout until the end
    def train1(self, graph, TAU, batch_size=10, iter_p=2):
        raise Exception('Deprecated')
        self._train(graph, TAU, batch_size, iter_p, False)

    # rollout only until leaf
    def train2(self, graph, TAU, batch_size=10, iter_p=2):
        self._train(graph, TAU, batch_size, iter_p, True)

    def test(self):
        result = [self.mcts.search(graph, 10) for graph in tqdm(self.test_graphs, unit='test_graph', leave=False)]
        #print(result)
        self.test_result.append(result)
        self._save_test_result()

    def _save_test_result(self):
        os.makedirs("log", exist_ok=True)
        with open("log/{}.pickle".format(self.filename), mode="wb") as f:
            pickle.dump(self.test_result, f)

    def _save_model(self):
        os.makedirs("model", exist_ok=True)
        torch.save(self.mcts.gnn.state_dict(), "model/{}.pth".format(self.filename))


class LoggingTrainer(MCTSTrainer):
    "Logs model per epoch"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = len(self.test_result)
        if isfile('plot_epochns/{}.pickle'.format(self.filename)):
            with open('plot_epochns/{}.pickle'.format(self.filename), 'rb') as file:
                self.plot_epochns = pickle.load(file)
        else:
            self.plot_epochns = [-1] * len(self.train_times)

    def _train(self, *args, **kwargs):
        super()._train(*args, **kwargs)
        self.epoch += 1

    def test(self, *args, **kwargs):
        super().test(*args, **kwargs)
        self.plot_epochns.append(self.epoch)
        os.makedirs('plot_epochns', exist_ok=True)
        with open('plot_epochns/{}.pickle'.format(self.filename), 'wb') as f:
            pickle.dump(self.plot_epochns, f)

    def _save_model(self):
        os.makedirs("model", exist_ok=True)
        torch.save(self.mcts.gnn.state_dict(), "model/{}_e{:03d}.pth".format(self.filename, self.epoch))

