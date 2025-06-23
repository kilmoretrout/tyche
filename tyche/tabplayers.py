# -*- coding: utf-8 -*-
import diskcache as dc
import numpy as np
import os
from .simulation import cards
from .query import QueryTree

# -*- coding: utf-8 -*-
import ctypes
import configparser
import sys

def get_strategy_from_config(config_file_path):
    """
    Reads the 'strategy' item from the 'settings' section of a config file.

    Args:
        config_file_path (str): The path to the configuration file.

    Returns:
        str: The value of the 'strategy' item, or None if not found.
    """
    config = configparser.ConfigParser()
    try:
        config.read(config_file_path)
        if 'settings' in config and 'strategy' in config['settings']:
            return config['settings']['strategy']
        else:
            print(f"Warning: 'strategy' not found in the 'settings' section of {config_file_path}")
            return None
    except configparser.Error as e:
        print(f"Error reading config file {config_file_path}: {e}")
        return None

# Get the absolute path to the directory containing this script
package_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the shared library
library_path = os.path.join(package_dir, 'lib', 'libhandindexer.so')

# Load the library
try:
    lib = ctypes.CDLL(library_path)
    # You can now use your library's functions, e.g., lib.some_function()
except OSError as e:
    print(f"Error loading library: {e}")
    sys.exit()
    # Handle the error appropriately

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
    def __init__(self, idir = get_strategy_from_config('settings.config')):
        idirs = [os.path.join(os.path.join(idir, 'strategy'), u) for u in ['preflop', 'flop', 'turn', 'river']]
        
        self.trees = [QueryTree(os.path.join(u, 'ball.tree'), u) for u in idirs]
        self.hand_indexers = [HandIndexer(1, [2]), 
                              HandIndexer(2, [2, 3]), 
                              HandIndexer(3, [2, 3, 1]), 
                              HandIndexer(4, [2, 3, 1, 1])]
        
        self.buckets = [np.load(os.path.join(idir, 'buckets/preflop.npz'))['lu'], 
                        np.load(os.path.join(idir, 'buckets/flop.npz'))['lu'], 
                        np.load(os.path.join(idir, 'buckets/river.npz'))['lu'], 
                        np.load(os.path.join(idir, 'buckets/turn.npz'))['lu']]
        starting_cache = dc.Cache(os.path.join(os.path.join(idir, 'strategy'), 'preflop/0'))

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