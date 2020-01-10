import os
import pickle

os.chdir('log')
for i in ('cinc', 'train100', 'train200', 'hardened', 'chardened', 'dynUCB'):
    if os.path.isfile('{}_0.pickle'.format(i)):
        with open('{}_0.pickle'.format(i), 'rb') as file2:
            b = pickle.load(file2)
        if i != 'train100':
            b = b[1:]
    else:
        b = []
    with open('{}.merge.pickle'.format(i), 'rb') as file1:
        a = pickle.load(file1)
    with open('{}_0.pickle'.format(i), 'wb') as file: pickle.dump(a+b, file)
    os.remove('{}.merge.pickle'.format(i))
    
