from utils.graph import read_graph
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse
from config import *
from mcts.mcts_node import MCTSNode
from mcts.mcts import MCTS
from gin.gin import *
from tqdm import *


idx = 0
gnn = GIN3(layer_num=6, feature=8, idx=idx)
gnn.load_state_dict(torch.load('model/m200_0.pth'))
gnn.to(device)
gnn.eval()
mcts = MCTS(gnn) #, performance=True
finish = False
stop_at_leaf = False

for graph_no in trange(10, unit='graph'):
    graph_filename = "random/100_250_{}".format(graph_no)
    graph = read_graph("data/" + graph_filename)
    degrees = [len(neighbors) for neighbors in graph.tmp]
    root_node = MCTSNode(graph.adj, mcts)
    statesList = []
    with open('nodewit.txt','a') as file:
        file.write('\ngraph: {}\n'.format(graph_filename))
    for i in trange(600, leave=False, unit='iteration'): 
        node = root_node
        v = -1
        states = [node]
        vertices = [i for i in range(graph.n)]
        while not finish:
            if node.is_end(): break
            if v==-1:
                raw_children = node.raw_children()
                if i<3 and graph_no==0:
                    orderByChildren = list(sorted(range(10), key=lambda i:raw_children[i]))
                    orderByDegrees = list(sorted(range(10), key=lambda i:degrees[i]))
                    print(orderByChildren, orderByDegrees) # debug
                if i%30==29:
                    #print(raw_children)
                    with open('nodewit.txt','a') as file:
                        print('Epoch',i,'children',raw_children, file=file)
                        orderByChildren = list(sorted(range(10), key=lambda i:raw_children[i]))
                        orderByDegrees = list(sorted(range(10), key=lambda i:degrees[i]))
                        if orderByChildren != orderByDegrees:
                            print('Not same order', file=file)
                            print('Not same order')
            v = node.best_child()
            realV = vertices[v]
            #print('{} chosen\n'.format(realV))
            if node.children[v] is None:
                env = MISEnv() if use_dense else MISEnv_Sparse()
                env.set_graph(node.graph)
                next_graph, r, done, info = env.step(v)
                node.children[v] = MCTSNode(next_graph, mcts, idx=v, parent=node)
                if stop_at_leaf:
                    finish = True
            node = node.children[v]
            for neighbor in graph.tmp[realV]:
                if neighbor in vertices: vertices.remove(neighbor)
            if realV in vertices: vertices.remove(realV)
            states.append(node)
        statesList.append(states)

        V = node.state_value()
        while node is not root_node:
            V += 1
            mcts.update_parent(node, V)
            node = node.parent
        mcts.root_max = max(mcts.root_max, V)

