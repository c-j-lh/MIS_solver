#!/usr/bin/env python3
from argparse import ArgumentParser
parser = ArgumentParser(description='train and save')
parser.add_argument('name', type=str)
parser.add_argument('--N-iter', type=float, default=-1)
parser.add_argument('--models', type=int, default=[0], nargs='+')
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--train-size', type=int, default=-1)
parser.add_argument('--test-size', type=int, choices=[-1, 10,100,200,500,1000,10000], default=-1)
parser.add_argument('--weights', type=int, nargs=2, default=(1,0))
parser.add_argument('--curricular',  action='store_true')
parser.add_argument('-c', type=str)
args = parser.parse_args()

import sys
from os import listdir
from os.path import isfile
from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from itertools import chain

from tqdm import trange, tqdm
import numpy as np
import torch

from config import device
from mcts.mcts import MCTS
from mcts.mcts_trainer import LoggingTrainer
from gin.gin import GIN3
from utils.graph import read_graph, generate_random_graph
from utils.timer import Timer
from utils.counter import Counter

retrain = True
def train_and_save(dynamic, models, epochs, base_train_size, test_size, weights, curricular, setup_name):
    "-1 or empty iterable for default values"
    if dynamic == -1: dynamic=1 if dynamic==-1 else dynamic
    if not models: models=range(4)
    if epochs==-1: epochs=100 # change to timed later
    if base_train_size==-1: base_train_size=100
    if test_size==-1: test_size=100
    if weights==-1: weights=(1,0)
    args = list(sorted(list(locals().items())))

    epochs = epochs if epochs!=-1 else int((200/dynamic)**(1/4)*200)
    base = 0.98 ** (100/epochs)
    print('setup_name: {}'.format(setup_name))

    with open('log/new_models.txt','a') as logfile:
        logfile.write('\nStarting training: {}.\nTime: {}\nsetup_name: {} (new dynamic)\n'.format(args, str(datetime.now())[:-7], setup_name))

    train_graphs = list(map(read_graph, ["data/frb/frb{}-{}".format(shape, i) for shape in ['30-15', '35-17', '40-19', '45-21', '50-23', '53-24', '56-25','59-26'] 

                                         # For hardened
                                         for i in range(4)]))

                                         # For chardened
                                         #for i in ('-2', '+2', '-1', '+1')]))
    train_sizes = list(chain(range(10, 50, 1), range(50, 100, 2), range(100, 275, 5)))

    for model_no in tqdm(models, unit="model_trained", leave=False):
        if retrain:
            for i in range(100):
                filename = 'model/' + setup_name.format('{}_e{:03d}'.format(model_no, i))
                if not isfile(filename):
                    break
        else:
            i = 0
        gnn = GIN3(layer_num=6, feature=8)
        if retrain and i:
            print(os.getcwd(), os.path.isfile('model/' + setup_name.format('{}_e{:03d}'.format(model_no, i-1))))
            filename = 'model/' + setup_name.format('{}_e{:03d}'.format(model_no, i-1))
            gnn.load_state_dict(torch.load(filename))
        gnn.to(device)
        if retrain and i:
            gnn.eval()
            print('Epochs trained:', i)

        model_name =  "{}_{}".format(setup_name,model_no)
        trainer = LoggingTrainer(gnn, (), model_name, dynamic, weights=weights, new=not retrain)
        trainer.test()

        # If non-hardened, non-curriculum2
        for epoch in trange(i, 80, unit='epoch', leave=True):

        # For hardened/chardened
        #for epoch, train_graph in enumerate(tqdm(train_graphs[i:], unit='epoch', leave=False)):

        # For curriculum2
        #for epoch, train_size in enumerate(tqdm(train_sizes[i:], unit='epoch', leave=False)):
            # Only for normal curriculum
            #train_size = int(epoch*5 + base_train_size)  

            # Just not for curriculum2
            train_size = base_train_size

            # Only for non-hardened
            train_graph = generate_random_graph(train_size, int(2.5*train_size))


            # Only for dynUCB
            if epoch==0: 
                trainer.mcts.weights = [10.0, 1]
            elif epoch==20: 
                trainer.mcts.weights = [9.0, 1] 
            elif epoch==40:
                trainer.mcts.weights = [8.0, 1] 
            elif epoch==60:
                trainer.mcts.weights = [7.0, 1]
            elif epoch==80:
                trainer.mcts.weights = [6.0, 1]
            elif epoch==100:
                trainer.mcts.weights = [5.0, 1]
            elif epoch==120:
                trainer.mcts.weights = [4.0, 1]

            trainer.train2(train_graph.adj, 10 * base ** epoch, iter_p=2)

            if epoch%5 == 4:
                trainer.test()

assert '.pth' not in args.name and '{}' not in args.name
train_and_save(dynamic=args.N_iter, models=args.models, epochs=args.epochs, base_train_size=args.train_size, test_size=args.test_size, weights=args.weights, curricular=args.curricular, setup_name=args.name)
#train_and_save(3)
#train_and_save(6)
#train_and_save(12)
#train_and_save(25)
#train_and_save(50)
#train_and_save(100)
