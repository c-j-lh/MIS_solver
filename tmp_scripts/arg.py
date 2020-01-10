from argparse import ArgumentParser

parser = ArgumentParser(description='Hello World')
parser.add_argument('--r-0')
args = parser.parse_args()

print(args.r_0)
