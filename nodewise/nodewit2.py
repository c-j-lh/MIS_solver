# not worth my brainpower
import pickle
with open('nodewit2.txt', 'r') as file:
    stores = []
    for graph_no in range(10):
        print(graph_no)
        store = {}
        line = file.readline() #graph: ...
        for current in range(29, 600, 30):
            store[current] = []
            for i in range(17):
                line = file.readline()
                tokens = line.split(' ')
                tokens = [token.replace('[','').replace(']','').replace(' ','').replace('\n','') for token in tokens]
                tokens = [token for token in tokens  if token!='']
                oldtokens = tokens[:]
                for token in oldtokens:
                    try:
                        print(int(token))
                    except:
                        tokens.append(token)
                #if i==16: tokens = tokens[:-1] # ending ]
                #elif i==0: tokens = tokens[2:] # starting 29 [
                #print(repr(line))
                try:
                    store[current].extend(list(map(float, tokens)))
                except: print(tokens); raise
            line = file.readline() # newline end of group
        line = file.readline() # newline end of graph
        stores.append(store)


with open('nodewit.pickle', 'wb') as file:
    pickle.dump(stores, file)
#  -5.9382516e-01
