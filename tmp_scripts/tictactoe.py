# so...
'''
build an interface. 
'''

from tqdm import tqdm

class Node:
    dp = {}
    def __init__(self, state, player, pbar=None):
        """State should be a list of 3x3=9 ints
        0 for empty, 1 for player 1, -1 for player -1
        """
        self.state = state
        self.player = player
        if pbar is None:
            pbar = tqdm(unit='function call')
        self.pbar = pbar

    def get_actions(self):
        "Returns a list of next nodes"
        return [Node(self.state[:index]+[self.player]+self.state[index+1:], -self.player, self.pbar) for index,val in enumerate(self.state) if val==0]

    def turn(self):
        return sum(cell!=0 for cell in self.state)

    def winner(self):
        totals = []
        for row in range(3): 
            totals.append(sum(self.state[3*row:3*(row+1)]))
        for col in range(3): 
            totals.append(sum(self.state[3*row+col] for row in range(3)))
        totals.append(sum(self.state[4*cnt] for cnt in range(3)))   # main diagonal
        totals.append(sum(self.state[2*cnt+2] for cnt in range(3))) # other diagonal
        for total in totals:
            if abs(total)==3: return total//3

        if sum(cell!=0 for cell in self.state)==9:
            return 2 # tie
        return 0
        

    def get_action(self):
        ans = self._get_action()
        self.pbar.close()
        return ans

    def _get_action(self):
        "Returns the (next best Node, reward)"
        if self in Node.dp:
            return Node.dp[self]
        self.pbar.update()
        #print('Calling',self)

        winner = self.winner()
        if winner:
            if abs(winner) == 1:
                return (self, 10*winner*self.player)
            return (self, self.turn())

        actions = self.get_actions()
        cases = [action._get_action() for action in actions]
        cases = [(action, -case[1]) for case,action in zip(cases,actions)] # invert the rewards
        case = max(cases, key=lambda pair:pair[1]) # max by reward
        Node.dp[self] = case
        return case

    def advance(self, x, y):
        "modifies state in-place"
        try:
            assert not self.state[x*3+y]
            self.state[x*3+y] = self.player
            self.player = -self.player
        except:
            raise ValueError('x,y is already occupied or out of bounds')
            
        
    def __str__(self):
        "Represents player 1 with X, player -1 with O"
        return '\n+---+\n' \
               + '\n'.join( \
                   ( 
                       '|' + ''.join(' XO'[self.state[3*i+j]] for j in range(3)) + '|'
                   ) for i in range(3)
                 ) \
               + '\n+---+\n' 

    def __repr__(self):
        "For printing in lists"
        return str(self)

    def __key(self):
        return tuple(self.state) + (self.player,)

    def __hash__(self):
        return hash(self.__key()) 

    def __eq__(self, other):
        return self.__key() == other.__key()

node = Node([0,-1,0, 0,1,0, 0,0,0], 1) # AI should win
#node = Node([0]*9, 1) # start board
print(node)
while not node.winner():
    if node.player == 1: # AI
        node, _ = node.get_action()
    else:
        while True:
            try:
                x,y = map(int,input("Enter coordinates: ").split())
                node.advance(x,y)
                break
            except ValueError:
                continue
    print(node)
print('You', 'win' if node.winner()==-1 else 'lose' if node.winner()==1 else 'tie')

