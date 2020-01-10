import os
import pickle

os.chdir('log')
filenames = sorted(os.listdir())
for filename in filenames:
    if not filename.endswith('.pickle'):
        continue

    try:
        with open(filename, 'rb') as file:
            lst = pickle.load(file)
            #print("{:40} | {:3d}".format(filename, len(lst)))
            print("{:40} | {}".format(filename, len(lst)))
    except Exception:
        print(filename + ' failed'); 
