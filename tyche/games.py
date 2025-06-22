# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
                
# each game state is a node
import struct

# assumes the starting stack size is 800 and that all players started with 800
def write_node2(r, actions, player_acting, c_idx, ifile, Tdata = None):
    ifile.write(len(actions).to_bytes(1, byteorder = 'big'))
    ifile.write(player_acting.to_bytes(1, byteorder = 'big'))
    ifile.write(r.to_bytes(1,  byteorder = 'big'))
    for ix, ii in enumerate(c_idx):
        ifile.write(struct.pack('>I', ii))
        
        # Get the corresponding action string
        action = actions[ix]  # Assuming 'actions' is indexed by c_idx entries

        # Write the length of the action string as an 8-bit number
        action_length = len(action)
        ifile.write(action_length.to_bytes(1, byteorder='big'))

        # Write each character of the action as an 8-bit number
        for char in action:
            ifile.write(ord(char).to_bytes(1, byteorder='big'))
        
    if Tdata:
        folded, stacks = Tdata
        
        _ = ''.join(list(map(str, np.zeros(8 - len(folded), dtype = np.uint8))))
        ifile.write(bitstring_to_bytes(''.join(list(map(str, [int(u) for u in folded]))) + _))
        
        for u in stacks:
            ifile.write(struct.pack('>H', int(800 - u)))
            
    return

class Player(object):
    def __init__(self, index = 0, stack = 8000):
        self.stack = stack
        self.index = index

class NLHoldemGame(object):
    def __init__(self, stack_size_bb = 100.0, bb = 8.0, sb = 4.0, n_players = 6,
                 bet_sizes = [[0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0] for u in range(3)], # as fraction of the pot
                 raise_sizes = [[1.0, 1.5, 2.0, 4.0, 3.0, 6.0, 8.0, 12.0] for u in range(4)],  # as fraction of last raise >= 1
                 max_raises = 6,
                 max_round_raises = [3, 2, 2, 2]):
    
        self.bet_sizes = bet_sizes
        self.raise_sizes = raise_sizes
        self.max_raises = max_raises
        self.max_round_raises = max_round_raises
        
        self.players = [Player(u, stack = stack_size_bb * bb) for u in range(n_players)]
        self.round = 0
        
        self.history = ''
        
        self.bb = bb
        self.sb = sb
        
        self.players[-2].stack -= sb # small blind
        self.players[-1].stack -= bb # big blind

        self.wagers = np.zeros((n_players, ))
        self.wagers[-1] = bb
        self.wagers[-2] = sb
        self.current = 0
        self.raise_count = 0
        self.folded = [False for u in range(n_players)]
        self.acted = [False for u in range(n_players)]
        self.children = []
        self.round_raise_count = 0
        
        self.pot = bb + sb
        
        self.o_stack = stack_size_bb * bb
        
        self.n_players = n_players
        
        self.wager = bb
        self.raise_amt = bb - sb
        
    def next_player(self):
        ii = (self.current + 1) % self.n_players
        
        _ = self.folded[ii]
        while _:
            ii = (ii + 1) % self.n_players
            _ = self.folded[ii]
            
        return ii 
    
    def possible_actions(self):
        actions = []
        
        if self.wagers[self.current] < self.wager:
            actions.append('f')
            actions.append('c')
            
        elif (self.wagers[self.current] == self.wager) and (not self.acted[self.current]):
            actions.append('c')
            
        if self.players[self.current].stack > self.wager and self.wager > 0:
            if self.raise_count < self.max_raises and self.round_raise_count < self.max_round_raises[self.round]:
                for u in self.raise_sizes[self.round]:
                    raise_by_amt = u * self.raise_amt
                    if raise_by_amt >= self.raise_amt:

                        new_wager = self.wager + raise_by_amt
                        wager_diff = new_wager - self.wagers[self.current]
                        if self.players[self.current].stack >= wager_diff:
                            actions.append('r({0:.4f})'.format(u))
            actions.append('r(-1.0000)')
        
        if (self.wager == 0 and self.players[self.current].stack != 0):
            for u in self.bet_sizes[self.round - 1] + [-1]:
                if u == -1:
                    actions.append('b({0:.4f})'.format(u))
                elif self.players[self.current].stack >= u * self.pot:
                    if u * self.pot >= self.bb:
                        actions.append('b({0:.4f})'.format(u))
                    
        return actions
        
    def terminal(self):
        ii = np.array(self.folded).astype(np.uint8)

        if np.sum(ii) == self.n_players - 1:
            return True
        
        if (self.round == 3) and (self.round_complete()):
            return True
        
        if self.allin():
            return True
        
        return False
    
    def allin(self):
        stacks = np.array([u.stack for u in self.players])
        ii = np.where(np.array(self.folded).astype(np.uint8) == 0)[0]

        return np.sum(stacks[ii]) == 0
    
    def round_complete(self):
        # has every non-folded player acted this round?
        acted = np.array(self.acted).astype(np.uint8) 
        ii = np.where(np.array(self.folded).astype(np.uint8) == 0)[0]
        
        x = np.all(acted[ii])
        
        # and the wagers of those players are equal
        y = np.var(self.wagers[ii]) == 0

        return (x and y)        

    def next_round(self):
        self.wagers = np.zeros((self.n_players, ))
        self.current = self.n_players - 3
        self.current = self.next_player()
        self.acted = [False for u in range(self.n_players)]
        self.wager = 0
        
        self.round += 1
        self.round_raise_count = 0
        self.raise_amt = 0.
        
        self.history += '|'

    def parse_action(self, action):
        ret = deepcopy(self)
        
        if 'f' in action:
            ret.folded[ret.current] = True
            #print('player {} folds...'.format(ret.current))
            
        elif 'c' in action:
            ret.players[ret.current].stack -= ret.wager - ret.wagers[ret.current]
            
            ret.pot += ret.wager - ret.wagers[ret.current]
            ret.wagers[ret.current] = ret.wager
            
        elif 'r' in action:
            r_amt = float(action.replace('(', '').replace(')', '')[1:])
            
            if r_amt == -1:
                wager_diff = ret.players[ret.current].stack
                r_by_amt = wager_diff
                
                new_wager = ret.wagers[ret.current] + wager_diff
            else:
                r_by_amt = ret.raise_amt * r_amt
                
                new_wager = ret.wager + r_by_amt
                wager_diff = new_wager - ret.wagers[ret.current]
            
            ret.players[ret.current].stack -= wager_diff
            ret.wagers[ret.current] = new_wager
            ret.wager = new_wager
            ret.pot += wager_diff
            ret.raise_amt = r_by_amt
            
            ret.raise_count += 1
            ret.round_raise_count += 1
                        
        elif 'b' in action:            
            b_amt = float(action.replace('(', '').replace(')', '')[1:])
            if b_amt == -1:
                wager_diff = ret.players[ret.current].stack
                new_wager = ret.wagers[ret.current] + wager_diff
                
                ret.raise_amt = new_wager
                
                ret.players[ret.current].stack -= wager_diff
                ret.wagers[ret.current] = new_wager
                ret.wager = new_wager
                ret.pot += wager_diff
            else:
                new_wager = b_amt * ret.pot
            
                ret.raise_amt = new_wager
            
                ret.players[ret.current].stack -= new_wager
                ret.wagers[ret.current] = new_wager
                ret.wager = new_wager
                ret.pot += new_wager
                    
        ret.acted[ret.current] = True
        ret.current = ret.next_player()
        ret.history += action
        
        if not ret.terminal():
            if ret.round_complete():                
                ret.next_round()
        
        return ret
    
import itertools
import networkx as nx
from collections import deque
import copy
import pickle
from .tabplayers import TabularStrategy
from scipy.interpolate import interp1d

N_BUCKETS = [169, 2048, 2048, 2012]

# for keeping track of reach and generating subtrees
class Simulator(object):
    def __init__(self, strat = None, reentrant_round = 0):
        self.reentrant_round = reentrant_round
        
        if strat is None:
            self.strat = TabularStrategy()
        else:
            self.strat = strat
        self.new_game()
        
    def new_game(self):
        self.deck = set(range(52))
        
        self.reach = np.zeros((6, 1326)) # log-likelihood reach for each possible pair (in later rounds some are ignored due to board)
        self.reachable = list(range(1326)) # all pairs are initially reachable
        self.history = ''
        self.game = NLHoldemGame(n_players = 6, bb = 1.0, sb = 0.5)
        self.board = []
        
        # map the possible hole cards to buckets
        self.indices = []
        ij = list(itertools.combinations(range(52), 2))

        for i, j in ij:
            self.indices.append(self.strat.buckets[0][self.strat.hand_indexers[0].index([i, j])])
            
        self.p = None
        
    def parse_action(self, action):
        # get the strategy
        deck = list(self.deck)
        p, idx, actions = self.strat.get(self.history, self.game.round, [deck[0], deck[1]] + self.board)
        # (bucket, action)
        p = p / p.sum(-1).reshape(-1, 1)

        self.p = p

        pix = self.game.current
        if action in ('f', 'c'):
            ii = actions.index(action)            
            
            self.reach[pix][self.reachable] += np.log(p[self.indices, ii] + 1e-8)
        else:
            frac = float(action.replace('r', '').replace('b', '').replace('(', '').replace(')', ''))
            
            x = [u for u in actions if (('r' in u) or ('b' in u))]
            x = [float(u.replace('(', '').replace(')', '').replace('r', '').replace('b', '')) for u in x]
            
            x = np.array(x)
            
            if self.game.raise_amt == 0:
                x[-1] = self.game.players[pix].stack / self.game.pot
            else:
                x[-1] = self.game.players[pix].stack / self.game.raise_amt

            f = interp1d(x, np.log(p[self.indices,-len(x):] + 1e-8))
            
            self.reach[pix][self.reachable] += f(frac)
            
        self.history += action
        self.game = self.game.parse_action(action)
    
    # upon dealing the board at the start of a new round
    # assumes self.game is in that round via self.parse_action
    def update_board(self, cards):
        self.board += cards
        
        self.deck = self.deck.difference(cards)
        
        # map the possible hole cards to buckets
        self.indices = []
        self.reachable = []
        
        ij = list(itertools.combinations(range(52), 2))

        for ix, (i, j) in enumerate(ij):
            if not ((i in self.board) or (j in self.board)):
                self.indices.append(self.strat.buckets[self.game.round][self.strat.hand_indexers[self.game.round].index([i, j] + self.board)])
                self.reachable.append(ix)
    
    def write_subtree(self, game, ofile, write_regrets = False):
        # save the reach
        np.savez(ofile.split('.')[0] + '_reach.npz', reach = self.reach,
                 folded = np.array(game.folded, dtype = np.uint8).reshape(1, -1),
                 board = np.array(self.board, dtype = np.uint8).reshape(1, -1))
        
        data = dict()
        
        n_players = 6
        
        ifile = open(ofile, 'wb')
        ifile.write(n_players.to_bytes(1, byteorder = 'big'))

        graph = nx.DiGraph()
        
        node_dict = dict()
        
        index = 0
        game.index = 0
        graph.add_node(index)
        index += 1
        n_action_nodes = 1
        
        todo = deque()
        todo.insert(0, game)
        
        print('computing subtree...')
        while len(todo) > 0:
            G = todo.pop()
            
            parent = G.index
            children = []
            
            actions = G.possible_actions()
            if not G.terminal():
                n_action_nodes += 1
                # put the children in the queue to be written and get their indices
                for ix, a in enumerate(actions):
                    Gc = G.parse_action(a)

                    key = (Gc.round, Gc.pot, Gc.wager, Gc.wagers[Gc.current], tuple(np.array(Gc.folded, dtype = np.uint8)), Gc.current, Gc.acted[Gc.current])
                    
                    if Gc.round >= self.reentrant_round:
                        if key in node_dict.keys():
                            Gc.index = node_dict[key]
                            graph.add_edge(parent, Gc.index, action = a)
                            
                        else:
                            Gc.index = copy.copy(index)
                            graph.add_edge(parent, copy.copy(index), action = a)
                            
                            if (not Gc.terminal()):
                                node_dict[key] = copy.copy(index)   
                            index += 1
                            
                            todo.insert(0, Gc)
                    else:
                        Gc.index = copy.copy(index)
                        graph.add_edge(parent, copy.copy(index), action = a)
                        index += 1
                        
                        todo.insert(0, Gc)
                        
            graph.nodes[parent]["player_acting"] = G.current
            graph.nodes[parent]["round"] = G.round
            graph.nodes[parent]["history"] = G.history
            if G.terminal():
                graph.nodes[parent]["folded"] = G.folded
                graph.nodes[parent]["pot"] = G.pot
                graph.nodes[parent]["stacks"] = [p.stack for p in G.players]
                
        print('have {} action nodes...'.format(n_action_nodes))
        print('have {} edges...'.format(graph.number_of_edges()))
        print('writing subtree...')        
        for ix, node in enumerate(sorted(graph.nodes)):
            edges = list(graph.out_edges(node))
            c_idx = [u[1] for u in edges]
                    
            actions = [graph[node][u]['action'] for u in c_idx]
            n_actions = len(actions)
            
            if "folded" in graph.nodes[node].keys():
                folded = graph.nodes[node]["folded"]
                pot = graph.nodes[node]["pot"]
                stacks = graph.nodes[node]["stacks"]
                
                Tdata = (folded, stacks)
                
                write_node2(graph.nodes[node]["round"], [], graph.nodes[node]["player_acting"], c_idx, ifile, Tdata)
            else:
                Tdata = None
                
                if write_regrets:
                    try:
                        node_history = graph.nodes[node]["history"]
                        
                        # dummy board
                        if graph.nodes[node]["round"] == 0:
                            board = list(range(2))
                        elif graph.nodes[node]["round"] == 1:
                            board = list(range(5))
                        elif graph.nodes[node]["round"] == 2:
                            board = list(range(6))
                        else:
                            board = list(range(7))
                            
                        r, _, _ = self.strat.get(node_history, graph.nodes[node]["round"], board)
                        data[str(ix) + 'r'] = r.astype(np.float32)
    
                        print('wrote regrets for {}...'.format(node_history))
                    except Exception as e:
                        data[str(ix) + 'r'] = np.zeros((N_BUCKETS[graph.nodes[node]["round"]], n_actions), dtype = np.float32)
                        pass
                
                write_node2(graph.nodes[node]["round"], actions, graph.nodes[node]["player_acting"], c_idx, ifile, Tdata)
        
        np.savez_compressed(ofile.replace('.gametree', '.npz'), **data)
        pickle.dump(graph, open(ofile.replace('.gametree', '.pkl'), 'wb'))
        
if __name__ == '__main__':
    import time

    t0 = time.time()

    print('loading strategy...')    
    sim = Simulator()
    
    print('parsing: ffffcc -> (some board)')
    sim.parse_action('f')
    sim.parse_action('f')
    sim.parse_action('f')
    sim.parse_action('f')
    sim.parse_action('c')
    sim.parse_action('c')
    
    sim.update_board([0, 4, 15])
    sim.parse_action('c')
    sim.parse_action('c')
    sim.update_board([51])
    
    t0 = time.time()
    sim.write_subtree(copy.deepcopy(sim.game), 'test.subtree')
    print(time.time() - t0)
    
    print(sim.reach[0])