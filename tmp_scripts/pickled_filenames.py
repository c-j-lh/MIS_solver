import pickle
import os

for filename in sorted(os.listdir()):
    if not filename.endswith('graphfilenames.pickle'):
        continue

    with open(filename, 'rb') as file:
        graphnames = pickle.load(file)
        print('{:50} | {}'.format(filename, graphnames))
