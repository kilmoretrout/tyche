# -*- coding: utf-8 -*-
import diskcache as dc
import numpy as np
import os
from .simulation import cards
from .query import QueryTree

# -*- coding: utf-8 -*-
import ctypes

lib = ctypes.CDLL('./libhandindexer.so')

MAX_ROUNDS = 4
SUITS = 4
CARDS = 52

# Define types
lib.hand_indexer_create.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8)]
lib.hand_indexer_create.restype = ctypes.c_void_p

lib.hand_indexer_destroy.argtypes = [ctypes.c_void_p]

lib.hand_indexer_index.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)]
lib.hand_indexer_index.restype = ctypes.c_bool

def get_rank(card):
    return card & 0xF

def get_suit(card):
    return card >> 4

class HandIndexer:
    def __init__(self, rounds, cards_per_round):
        self.rounds = rounds
        self.cards_per_round = (ctypes.c_uint8 * rounds)(*cards_per_round)
        self.handle = lib.hand_indexer_create(rounds, self.cards_per_round)
        if not self.handle:
            raise RuntimeError("Failed to initialize hand indexer")

    def __del__(self):
        if self.handle:
            lib.hand_indexer_destroy(self.handle)

    def index(self, cards):
        """cards: list of 8-bit integers representing cards"""
        assert len(cards) == sum(self.cards_per_round), "Invalid number of cards"
        card_array = (ctypes.c_uint8 * len(cards))(*cards)
        index_out = ctypes.c_uint32()
        success = lib.hand_indexer_index(self.handle, card_array, ctypes.byref(index_out))
        if not success:
            raise RuntimeError("Indexing failed")
        return index_out.value

class TabularStrategy(object):
    def __init__(self, idir = '/mirror/src/cfrrust/games/strategy'):
        idirs = [os.path.join(idir, u) for u in ['preflop', 'flop', 'turn', 'river']]
        
        self.trees = [QueryTree(os.path.join(u, 'ball.tree'), u) for u in idirs]
        self.hand_indexers = [HandIndexer(1, [2]), 
                              HandIndexer(2, [2, 3]), 
                              HandIndexer(3, [2, 3, 1]), 
                              HandIndexer(4, [2, 3, 1, 1])]
        
        self.buckets = [np.load('/mirror/src/cfrrust/rust_buckets/preflop.npz')['lu'], 
                        np.load('/mirror/src/cfrrust/rust_buckets/flop.npz')['lu'], 
                        np.load('/mirror/src/cfrrust/rust_buckets/turn.npz')['lu'], 
                        np.load('/mirror/src/cfrrust/rust_buckets/river.npz')['lu']]
        starting_cache = dc.Cache(os.path.join(idir, 'preflop/0'))

        self.start = starting_cache['']
        
    def get(self, history, street, board):        

        hi = self.hand_indexers[street]
        
        idx = hi.index(board)
        
        idx = self.buckets[street][idx]

        if history != '':
            q = self.trees[street]
            p, actions = q.retrieve(history)
        else:
            p, actions = self.start
        
        return p, idx, actions
    
if __name__ == '__main__':
    import itertools
    
    hi = HandIndexer(1, [2])
    
    ij = list(itertools.combinations(range(52), 2))
    
    for k in range(len(ij)):
        i, j = ij[k]
        print(cards[i], cards[j], hi.index([i, j]))