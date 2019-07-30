import numpy as np
from config import use_dense

def make_adj_set_sparse(adj):
    n, _ = adj.shape
    row = adj.row
    col = adj.col
    m = row.size
    ss = [set() for _ in range(n)]
    for i in range(m):
        a = int(row[i])
        b = int(col[i])
        if a < b:
            ss[a].add(b)
            ss[b].add(a)
    return ss

def make_adj_set(adj):
    if not use_dense:
        return make_adj_set_sparse(adj)
    n, _ = adj.shape
    ss = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                ss[i].add(j)
                ss[j].add(i)
    return ss

def randomplay(ss):
    n = len(ss)
    vs = [i for i in range(n)]
    np.random.shuffle(vs)
    ng = set()
    ret = 0
    for v in vs:
        if not v in ng:
            ng.add(v)
            ng |= ss[v]
            ret += 1
    return ret

def randomplay_tsp(graph, dist_from_prev, dist_to_start):
    n = graph.shape[0]
    vs = [i for i in range(n)]
    np.random.shuffle(vs)
    ret = 0
    ret -= dist_from_prev[vs[0]]
    for i in range(1, n):
        ret -= graph[vs[i - 1]][vs[i]]
    ret -= dist_to_start[vs[-1]]
    return ret
