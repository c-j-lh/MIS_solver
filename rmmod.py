#!/usr/bin/env python3
from argparse import ArgumentParser
from os import remove
from os.path import isfile

parser = ArgumentParser(description='Delete model')
parser.add_argument('model', type=str, help='model to delete')
args = parser.parse_args()

filenames = []
for filename in ('model/{}_{}.pth', 'log/{}_{}.pickle', 'train_time/{}_{}.pickle', 'ctrain_time/{}_{}.pickle'):
    for model_no in range(10):
        if isfile(filename.format(args.model, model_no)):
            filenames.append(filename.format(args.model, model_no))
removable = bool(filenames)

with open('log/new_models.txt', 'r') as file:
    skip = 0
    lines = []
    for line in file:
        if "setup_name: {} (new dynamic)".format(args.model) == line.strip():
            print('Lines detected:', end='\n\t')
            print(*lines[-2:], line, sep='\t')
            lines = lines[:-2]
            removable = True
            continue
        lines.append(line)

if filenames:
    print('Files detected:')
    for filename in filenames:
        print('\t', filename)
else:
    print('No filenames detected')


if removable and input('\nRemove the above? [y/n]   ').lower()[:1] == 'y':
    for filename in filenames:
        remove(filename)
    with open('log/new_models.txt', 'w') as file:
        file.write(''.join(lines))

