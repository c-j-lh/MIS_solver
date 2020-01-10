import os

assert 'data/frb' in os.getcwd()

for filename in sorted(os.listdir()):
    if len(filename) == len('frb35-17-1'):
        continue

    with open(filename, 'r') as file:
        start = file.readline()
        n, m = map(int, start.split())
        edges = [map(str, (n-1, m))]
        for line in file:
            a, b = map(int, line.split())
            a -= 1
            b -= 1
            edges.append(map(str, (a, b)))

    with open(filename, 'w') as file:
        file.write(start)
        for edge in edges:
            file.write(' '.join(edge) + '\n')
            
