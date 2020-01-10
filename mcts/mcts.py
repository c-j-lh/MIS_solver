import time
import numpy as np
from config import use_dense
import torch
from tqdm import tqdm, trange
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse
from mcts.mcts_node import MCTSNode
from utils.graph import read_graph
from utils.timer import Timer
from utils.gnnhash import GNNHash
from utils.nodehash import NodeHash

EPS = 1e-30  # cross entropy loss: pi * log(EPS + p) (in order to avoid log(0))

class MCTS:
    def __init__(self, gnn, performance=False, dynamic=-1, weights=(5,1)):
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.003, weight_decay=1e-6)
        self.gnn = gnn
        self.nodehash = NodeHash(5000)
        self.gnnhash = GNNHash()
        # max reward of root in rollout
        self.root_max = 0
        self.performance = performance
        self.dynamic = dynamic
        self.counts = []

    # update Q(s,a), N(s,a) of parent
    def update_parent(self, node, V):
        par = node.parent
        normalized_V = par.normalize_reward(V)
        if par.visit_cnt[node.idx] == 0:
            # Q is an initial value
            par.Q[node.idx] = normalized_V
        else:
            self.update_Q(par, normalized_V, node.idx, method="mean")
        par.visit_cnt[node.idx] += 1

    def update_Q(self, node, V, idx, method):
        if method == "mean": # the case
            node.Q[idx] = (node.Q[idx] * node.visit_cnt[idx] + V) / (node.visit_cnt[idx] + 1)
        elif method == "max":
            node.Q[idx] = max(node.Q[idx], V)
        elif method == "min":
            node.Q[idx] = min(node.Q[idx], V)
        else:
            assert False

    def rollout(self, root_node, stop_at_leaf=False):
        node = root_node
        v = -1
        finish = False
        while not finish:
            if node.is_end(): break
            v = node.best_child()
            if node.children[v] is None:
                env = MISEnv() if use_dense else MISEnv_Sparse()
                env.set_graph(node.graph)
                next_graph, r, done, info = env.step(v)
                node.children[v] = MCTSNode(next_graph, self, idx=v, parent=node)
                if stop_at_leaf:
                    finish = True
            node = node.children[v]

        # backpropagate V
        V = node.state_value()
        while node is not root_node:
            V += 1
            self.update_parent(node, V)
            node = node.parent
        self.root_max = max(self.root_max, V)
        return int(V)

    # return improved pi by MCTS
    def get_improved_pi(self, root_node, TAU, iter_p=2, stop_at_leaf=False):
        assert not root_node.is_end()
        self.root_max = 0
        n, _ = root_node.graph.shape
        n_iter = int(self.dynamic * min(500, max(50, n * iter_p)))
        print('\ndynamic:', self.dynamic, 'n:', n, 'iter_p:', iter_p, 'n_iter', n_iter, '\n')
        self.counts[-1].append([])
        for i in trange(n_iter, leave=False, unit='iteration'):
        #for i in range(min(500, max(50, n * iter_p))): #original
            self.counts[-1][-1].append(self.rollout(root_node, stop_at_leaf=stop_at_leaf))
        return root_node.pi(TAU)

    def train(self, graph, TAU, batch_size=10, iter_p=2, stop_at_leaf=False):
        self.counts.append([])
        self.gnnhash.clear()
        mse = torch.nn.MSELoss()
        env = MISEnv() if use_dense else MISEnv_Sparse()
        env.set_graph(graph)

        graphs = []
        actions = []
        pis = []
        means = []
        stds = []
        done = False
        pbar = tqdm(leave=False, unit="node_added_to_MIS")
        while not done:
            n, _ = graph.shape
            node = MCTSNode(graph, self)
            means.append(node.reward_mean)
            stds.append(node.reward_std)
            print('get_improved_pi called')
            pi = self.get_improved_pi(node, TAU, iter_p=iter_p, stop_at_leaf=stop_at_leaf)
            action = np.random.choice(n, p=pi)
            graphs.append(graph)
            actions.append(action)
            pis.append(pi)
            graph, reward, done, info = env.step(action)
            pbar.update(1)

        T = len(graphs)
        idxs = [i for i in range(T)]
        np.random.shuffle(idxs)
        i = 0
        while i < T:
            size = min(batch_size, T - i)
            self.optimizer.zero_grad()
            loss = torch.Tensor([0])
            for j in range(i, i + size):
                idx = idxs[j]
                Timer.start('gnn')
                p, v = self.gnn(graphs[idx], True)
                Timer.end('gnn')
                n, _ = graphs[idx].shape
                # normalize z with mean, std
                z = torch.tensor(((T - idx) - means[idx]) / stds[idx])
                loss += mse(z, v[actions[idx]]) - (torch.tensor(pis[idx]) * torch.log(p + EPS)).sum()
            loss /= size
            loss.backward()
            self.optimizer.step()
            i += size

    # rollout iter_num times
    def search(self, graph, iter_num=10):
        root_node = MCTSNode(graph, self)
        ans = []
        for i in trange(iter_num, leave=False, unit='rollout iteration'):
            r = self.rollout(root_node)
            if self.performance: print(r)
            ans.append(r)
        return ans

    # rollout iter_num times
    def search_for_exp(self, graph, time_limit=600, min_iter_num=100):
        now = time.process_time()
        root_node = MCTSNode(graph, self)
        ans = []
        cnt = 0
        pbar = tqdm(leave=False, unit="rollout_{}_iteration".format(self.gnn.idx))
        while time.process_time() - now < time_limit or cnt == 0:
            r = self.rollout(root_node)
            pbar.update(1)
            ans.append(r)
            cnt += 1
        pbar.close()
        return ans

    # get improved pi for every action (by rolling out until end)
    def best_search1(self, graph, TAU=0.1, iter_p=1):
        self.gnnhash.clear()
        env = MISEnv() if use_dense else MISEnv_Sparse()
        env.set_graph(graph)

        ma = 0
        reward = 0
        done = False
        while not done:
            n, _ = graph.shape
            node = MCTSNode(graph, self)
            pi = self.get_improved_pi(node, TAU, iter_p=iter_p)
            ma = max(ma, self.root_max + reward)
            action = np.random.choice(n, p=pi)
            graph, reward, done, info = env.step(action)
        return ma, reward

    # get improved pi for every action (by rolling out only until leaf)
    def best_search2(self, graph, TAU=0.1, iter_p=1):
        self.gnnhash.clear()
        env = MISEnv() if use_dense else MISEnv_Sparse()
        env.set_graph(graph)

        reward = 0
        done = False
        while not done:
            n, _ = graph.shape
            node = MCTSNode(graph, self)
            pi = self.get_improved_pi(node, TAU, iter_p=iter_p, stop_at_leaf=True)
            action = np.random.choice(n, p=pi)
            graph, reward, done, info = env.step(action)
        return reward

    # use only p
    def policy_search(self, graph):
        env = MISEnv() if use_dense else MISEnv_Sparse()
        env.set_graph(graph)

        reward = 0
        done = False
        while not done:
            n, _ = graph.shape
            with torch.no_grad():
                p, v = self.gnn(graph)
            action = np.random.choice(n, p=p.detach().numpy())
            graph, reward, done, info = env.step(action)
        return reward

    # used only v
    def greedy_v_search(self, graph):
        env = MISEnv() if use_dense else MISEnv_Sparse()
        env.set_graph(graph)

        reward = 0
        done = False
        while not done:
            n, _ = graph.shape
            with torch.no_grad():
                p, v = self.gnn(graph)
            action = v.detach().numpy().argmax()
            graph, reward, done, info = env.step(action)
        return reward
