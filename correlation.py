from utils.graph import read_graph
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse
from config import *
from mcts.mcts_node import MCTSNode
from mcts.mcts import MCTS
from gin.gin import *

idx = 0
gnn = GIN3(layer_num=6, feature=8, idx=idx)
gnn.load_state_dict(torch.load('model/m200_0.pth'))
gnn.to(device)
gnn.eval()
mcts = MCTS(gnn) #, performance=True

for graph_no in range(10):
    graph_filename = "random/10_25_{}".format(graph_no)
    graph = read_graph("data/" + graph_filename)

    node = MCTSNode(graph.adj, mcts)
    v = node.best_child()
    print(v)
    print([(i,len(graph.tmp[i])) for i in range(graph.n)])
    print()

#V = node.state_value()
#while node is not root_node:
#    V += 1
#    self.update_parent(node, V)
#    node = node.parent
#self.root_max = max(self.root_max, V)
#return V

#return max(result)
