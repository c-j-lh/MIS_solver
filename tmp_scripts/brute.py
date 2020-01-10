from igraph import Graph

with open('data/random/100_250_0','r') as file:
    n,e = map(int, file.readline().split())
    graph = Graph()
    graph.add_vertices(n)
    print('n,e',n,e)
    graph.add_edges(tuple(map(int,line.split())) for line in file.readlines())
    print('Loaded')
    size = len(graph.largest_independent_vertex_sets()[0])
