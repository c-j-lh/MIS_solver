#!/usr/bin/env python3
from argparse import ArgumentParser
parser = ArgumentParser(description='train and save')
parser.add_argument('name', type=str)
parser.add_argument('--N_iter', type=float, default=-1)
parser.add_argument('--models', type=int, default=[0], nargs='+')
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--train_size', type=int, default=-1)
parser.add_argument('--test_size', type=int, choices=[-1, 10,100,200,500,1000,10000], default=-1)
parser.add_argument('--weights', type=int, nargs=2, default=(1,0))
parser.add_argument('--curricular',  action='store_true')
parser.add_argument('-c', type=str)
args = parser.parse_args()


from config import device
import numpy as np
import torch
from utils.graph import read_graph, generate_random_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer import LoggingTrainer
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter
from tqdm import trange, tqdm
from datetime import datetime
from os import listdir
from contextlib import redirect_stderr, redirect_stdout

def train_and_save(dynamic, models, epochs, base_train_size, test_size, weights, curricular, setup_name):
    "-1 or empty iterable for default values"
    if dynamic == -1: dynamic=1 if dynamic==-1 else dynamic
    if not models: models=range(4)
    if epochs==-1: epochs=600 # change to timed later
    if base_train_size==-1: base_train_size=100
    if test_size==-1: test_size=100
    if weights==-1: weights=(1,0)
    args = list(sorted(list(locals().items())))

    epochs = epochs if epochs!=-1 else int((200/dynamic)**(1/4)*200)
    base = 0.98**(100/epochs)
    with open('log/new_models.txt','r') as logfile:
        saved = logfile.read()
    if sum(('{}_{}.pth'.format(setup_name, model_no) in listdir('model')) \
              or ('{}_{}.pickle'.format(setup_name, model_no) in listdir('log')) \
              or (setup_name in listdir('example_log')) \
              or ('setup_name: {} (new dynamic)'.format(setup_name) in saved) for model_no in range(100)):
        raise Exception('model {} already exists'.format(setup_name))
    print('setup_name: {}'.format(setup_name))

    with open('log/new_models.txt','a') as logfile:
        logfile.write('\nStarting training: {}.\nTime: {}\nsetup_name: {} (new dynamic)\n'.format(args, str(datetime.now())[:-7], setup_name))

    for model_no in tqdm(models, unit="model_trained", leave=False):
        if __name__ == "__main__":
            gnn = GIN3(layer_num=6)
            gnn.to(device)
            model_name =  "{}_{}".format(setup_name,model_no)
            # default test graphs
            trainer = LoggingTrainer(0, gnn, (), model_name, dynamic, weights=weights)
            trainer.test()

            train_graphs = list(map(read_graph, ["data/frb/frb{}-{}.mis".format(shape, i) for shape in ['30-15', '35-17', '40-19', '45-21', '50-23', '53-24', '56-25','59-26'] 
                    for i in range(5)]))
                    #for i in ('-2', '+2', '-1', '+1', '')]))

            for epoch in trange(40, unit='epoch', leave=True):
            #for epoch, train_graph in enumerate(tqdm(train_graphs, unit='epoch', leave=False)):
                train_size = base_train_size #int(epoch*5 + base_train_size)
                train_graph = generate_random_graph(train_size, int(2.5*train_size))
                trainer.train2(train_graph.adj, 10 * base ** epoch, iter_p=2)
                if epoch==0: 
                    trainer.mcts.weights = [12.0*0.9**0,1]
                elif epoch==20: 
                    trainer.mcts.weights = [12.0*0.9**1,1]
                elif epoch==40:
                    trainer.mcts.weights = [12.0*0.9**3,1]
                elif epoch==60:
                    trainer.mcts.weights = [12.0*0.9**6,1]
                elif epoch==80:
                    trainer.mcts.weights = [12.0*0.9**10,1]
                elif epoch==100:
                    trainer.mcts.weights = [12.0*0.9**15,1]
                elif epoch==120:
                    trainer.mcts.weights = [12.0*0.9**21,1]

                trainer.test()
                #if i%20==19:

train_and_save(dynamic=args.N_iter, models=args.models, epochs=args.epochs, base_train_size=args.train_size, test_size=args.test_size, weights=args.weights, curricular=args.curricular, setup_name=args.name)
#train_and_save(3)
#train_and_save(6)
#train_and_save(12)
#train_and_save(25)
#train_and_save(50)
#train_and_save(100)
