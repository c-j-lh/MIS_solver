from random import shuffle

from tqdm import trange, tqdm

from utils.graph import read_graph, write_graph

add_nodes = False
remove_nodes = False

# Add nodes
if add_nodes:
    for shape in ['30-15', '35-17', '40-19', '45-21', '50-23', '53-24', '56-25','59-26']:
        g = read_graph('data/frb/frb{}-0.mis'.format(shape))
        choices = set()
        s = set((i,j) for i in range(1, g.n) for j in range(i, g.n))
        for node, neighbors in enumerate(g.tmp):
            for neighbor in neighbors:
                if neighbor > node:
                    choices.add((node, neighbor))
        s.difference_update(choices)

        write_graph(g, 'data/frb/frb{}-0+{}'.format(shape, 0))
        s = list(s)
        shuffle(s)
        for i in trange(1, 5):
            interval = len(choices) // 10
            for a, b in s[(i-1) * interval : i * interval]:
                g.tmp[a].add(b)
                g.tmp[b].add(a)
            g.build()
            tqdm.write('n={} m={}'.format(g.n, g.m))
            write_graph(g, 'data/frb/frb{}-0+{}'.format(shape, i))

# Remove nodes
if remove_nodes:
    for shape in ['30-15', '35-17', '40-19', '45-21', '50-23', '53-24', '56-25','59-26']:
        g = read_graph('data/frb/frb{}-0.mis'.format(shape))
        choices = []
        for node, neighbors in enumerate(g.tmp):
            for neighbor in neighbors:
                if neighbor > node:
                    choices.append((node, neighbor))

        write_graph(g, 'data/frb/frb{}-0-{}'.format(shape, 0))
        shuffle(choices)
        for i in trange(1, 5):
            interval = len(choices) // 10
            for a, b in choices[(i-1) * interval : i * interval]:
                g.tmp[a].remove(b)
                g.tmp[b].remove(a)
            g.build()
            write_graph(g, 'data/frb/frb{}-0-{}'.format(shape, i))

                    #train_graphs = list(map(read_graph, ["data/frb/frb{}-0-{}.mis".format(shape, i) for shape in ['30-15', '35-17', '40-19', '45-21', '50-23', '53-24', '56-25','59-26'] for i in range(1)]))
