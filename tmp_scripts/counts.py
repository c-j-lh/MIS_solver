import os
import pickle

import matplotlib.pyplot as plt
plt.ion()

os.chdir('train_time')
filenames = sorted(os.listdir())
lsts = []
markers = ".,ov^<>1234s"
for filename, marker in zip(filenames, markers):
    if not filename.endswith('.pickle'):
        continue

    try:
        if os.path.isfile('../counts/' + filename):
            with open('../counts/' + filename, 'rb') as file:
                lst = pickle.load(file)
                #print("{:40} | {:3d}".format(filename, len(lst)))
        else:
            lst = []
        #lsts.append(lst)

        with open(filename, 'rb') as file:
            times = pickle.load(file)
        print("{:40} | {:2d} epochs | {}".format(filename, len(lst),
                                                 [sum(j for i in epoch for j in i) for epoch in lst]))
        print(' '*40 +"             | {}\n".format([round(time) for time in times]))
        if lst:
            plt.plot([sum(j for i in epoch for j in i) for epoch in lst], times, 'o',
                     label=filename.split('.pickle')[0], marker=marker)
    except Exception:
        print(filename + ' badly formatted'); 
        raise
plt.legend()
plt.xlabel('Total number of iterations (in an epoch)')
plt.ylabel('Process time (time.process_time())/s (per epoch)')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('#iterations against time per epoch')
