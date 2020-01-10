import os
import sys

import torch

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from gin.gin import GIN3
from config import device

os.chdir('model')
errs = []
for filename in sorted(os.listdir()):
    try:
        #gnn = GIN3(layer_num=6, feature=8)
        torch.load(filename)
        #gnn.load_state_dict(torch.load(filename))
        #gnn.to(device)
        #gnn.eval()
        print('{:20} succeeded'.format(filename))
    except Exception as e:
        print('{:20} failed'.format(filename))
        errs.append(e.args[0])

for i in errs: print(i)
    
