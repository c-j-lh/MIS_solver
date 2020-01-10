import os

os.chdir('data/frb')
for filename in os.listdir():
    if len(filename) == len('frb35-17-0-0.mis'):
        print(filename.split('.mis')[0])
        os.rename(filename, './'+filename.split('.mis')[0])
