import os
import pickle

for fname in sorted(os.listdir('log')):
    if not fname.endswith('.pickle'):
        continue
    with open('log/' + fname, 'rb') as file:
        log = pickle.load(file)
    if len(log) < 30:
        print('\n{:40} --> {:3}'.format(fname, len(log)), end='')
        continue
    results = log[30]
    print('\n{:40} | {:3} | '.format(fname, len(results)), end='')
    for result in results[:25]:
        print('{:4} '.format(int(sum(result)/len(result))), end='')
    
