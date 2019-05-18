import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts.mcts_node import MCTSNode, INF
from utils.graph import read_graph

EPS = 1e-30  # cross entropy lossをpi * log(EPS + p)で計算 (log(0)回避)

class MCTS:
    def __init__(self, gnn):
        MCTSNode.set_gnn(gnn)
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
        self.gnn = gnn

    # parantのQ(s,a), N(s,a)を更新
    def update_parant(self, node, V):
        par = node.parent
        if par.Q[node.idx] >= INF:
            # Qが初期値から一度も更新されていない
            par.Q[node.idx] = V
        else:
            self.update_Q(par, V, node.idx, method="mean")
        par.visit_cnt[node.idx] += 1

    def update_Q(self, node, V, idx, method="mean"):
        if method == "mean":
            node.Q[idx] = (node.Q[idx] * node.visit_cnt[idx] + V) / (node.visit_cnt[idx] + 1)
        elif method == "max":
            node.Q[idx] = max(node.Q[idx], V)
        elif method == "min":
            node.Q[idx] = min(node.Q[idx], V)
        else:
            assert False

    def rollout(self, root_node, stop_at_leaf=True):
        node = root_node
        v = -1
        finish = False
        while not finish:
            if node.is_end(): break
            v = node.best_child()
            if node.children[v] is None:
                env = MISEnv()
                env.set_graph(node.graph)
                next_graph, r, done, info = env.step(v)
                node.children[v] = MCTSNode(next_graph, idx=v, parent=node)
                if stop_at_leaf:
                    finish = True
            node = node.children[v]

        # Vをrootに向かって伝播させていく
        V = node.state_value()
        while node is not root_node:
            V += 1
            self.update_parant(node, V)
            node = node.parent
        return V

    # MCTSによって改善されたpiを返す
    def get_improved_pi(self, graph, iter_p=5):
        root_node = MCTSNode(graph)
        assert not root_node.is_end()
        for i in range(graph.shape[0] * iter_p):
            self.rollout(root_node)
        return root_node.pi()

    def train(self, graph, batch_size=10):
        mse = torch.nn.MSELoss()
        env = MISEnv()
        env.set_graph(graph)

        graphs = []
        actions = []
        pis = []
        done = False
        while not done:
            n, _ = graph.shape
            pi = self.get_improved_pi(graph)
            action = np.random.choice(n, p=pi)
            graphs.append(graph)
            actions.append(action)
            pis.append(pi)
            graph, reward, done, info = env.step(action)

        T = len(graphs)
        for i in range(T):
            self.optimizer.zero_grad()
            loss = torch.Tensor([0])
            for batch in range(batch_size):
                idx = np.random.randint(T)
                p, v = MCTSNode.gnn(graphs[idx])

                n, _ = graphs[idx].shape
                z = torch.as_tensor(T - idx)
                # MSE(z,v)の方は適当にz^2で割ってスケールを合わせておく
                loss += mse(z, v[actions[idx]]) / z.pow(2) - (torch.as_tensor(pis[idx]) * torch.log(p + EPS)).sum()
            loss /= batch_size
            loss.backward()
            self.optimizer.step()

    def search(self, graph, iter_num=100):
        root_node = MCTSNode(graph)
        ans = []
        for i in range(iter_num):
            ans.append(self.rollout(root_node, stop_at_leaf=False))
        return ans