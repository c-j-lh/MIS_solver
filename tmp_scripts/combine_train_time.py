import pickle
import os

os.chdir('ctrain_time')
filenames = sorted(os.listdir())
for filename in filenames:
    if '_0' not in filename:
        continue
    filename2 = filename[:filename.index('_0')] + '_{}.pth_0.pickle'
    if filename2 in filenames:
        with open(filename, 'rb') as file1, \
             open(filename[:filename.index('_0')] + '_{}.pth_0.pickle', 'rb') as file2:
             a, b = pickle.load(file1), pickle.load(file2)
        out = a + [ctime + a[-1] for ctime in b]
        with open(filename, 'wb') as file: pickle.dump(out, file)
        os.remove(filename2)
#os.remove('dynamic.pickle')
