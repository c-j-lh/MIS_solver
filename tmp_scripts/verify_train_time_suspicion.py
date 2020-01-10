import os
import pickle

import matplotlib.pyplot as plt
#plt.ion()

setup_names = ['novatrain100', 'novatrain200', 'novatrain300', 'novacinc', 'newtrain100', 'newtrain200']
#setup_names = ['m.20', 'm.30', 'm.40', 'newm.5', 'train100']
setup_names = ['c3', 'c4', 'novatrain100', 'c6', 'newm.5', 'train100']

setup_names = [name + '_0.pickle' for name in setup_names]

#for setup_name in sorted(os.listdir('ctrain_time')):
for setup_name in setup_names:
    with open('ctrain_time/{}'.format(setup_name), 'rb') as file:
        time = [0.0] + [i/3600 for i in pickle.load(file)]
    plt.plot(time, label=setup_name.split('.pickle')[0])

plt.xlabel('Number of Epochs')
plt.ylabel('Time/h')
plt.legend()
plt.show()
