import os

os.chdir('model')
filenames = sorted(os.listdir())
for filename in filenames:
    deletend = '_{}.pth' 
    print(filename)
    if deletend not in filename:
        continue
    print(filename)
    os.rename(filename, filename.replace(deletend, ''))

