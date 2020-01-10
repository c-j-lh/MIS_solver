import re
import matplotlib.pyplot as plt
plt.ion()

for i, part in enumerate(('0_10', '237_247')):
    with open('new{}.txt'.format(part), 'r') as file:
        ls = []
        for line in file:
            print(line)
            ls.append([])
            for token in re.split(' |,|\[|]',line):
                if not token: continue
                print('\t',token)
                try:
                    ls[-1].append(float(token))
                except:
                    pass
    for epoch, epochresults in enumerate(ls):
        ls[epoch].append(sum(epochresults[1:])/10)

    with open('new{}.csv'.format(part),'w') as file:
        ls[0:0] = [['max']+['test {}'.format(i) for i in range(10)]+['average']]
        for line in ls:
            file.write(','.join(map(str,line))+'\n')

    plt.figure(i)
    for epoch,line in enumerate(ls[1:]):
        plt.plot([epoch]*10,line[1:11], 'o')
        plt.xlabel('epoch')
        plt.ylabel('result (graph size)')
        plt.title('Ran for 10 epochs on the test graph, with tau as if it were epochs {}. After each epoch, tested 10 times.'.format(part))
    print(ls)
