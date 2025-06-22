# -*- coding: utf-8 -*-
import numpy as np
import pickle
from .simulation import history_to_vec
import diskcache as dc
import os

possible_actions = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -1.0]

def format_ra(regrets, actions):
    ii = np.where(np.sum(regrets, -1) == 0.)[-1]

    regrets[ii] = np.ones(regrets.shape[1]) / regrets.shape[1]
    regrets = regrets / np.sum(regrets, -1).reshape(-1, 1)
    
    y_ = np.zeros((regrets.shape[0], len(possible_actions) + 2))
    action_ii = []
    
    for a in actions:
        if a == 'f':
            action_ii.append(0)
        elif a == 'c':
            action_ii.append(1)
        else:
            action_ii.append(possible_actions.index(float(a.replace('(', '').replace(')', '').replace('r', '').replace('b', ''))) + 2)

    for j in range(regrets.shape[0]):
        y_[j, action_ii] = regrets[j]
        
    return y_

class QueryTree(object):
    def __init__(self, ifile, idir):
        self.keys = pickle.load(open(ifile, 'rb'))
        self.idir = idir
    
    def retrieve(self, history):
        h, L = history_to_vec(history)
        h = h.flatten().reshape(1, -1)
        
       
        nbrs, keys_, idir = self.keys[L]
        idir = idir.split('/')[-1]
        
        cache = dc.Cache(os.path.join(self.idir, idir))
        
        d, ii = nbrs.kneighbors(h)
        i, j = ii[0]
        di, dj = d[0]
        ki = keys_[i]
        kj = keys_[j]
        
        if (di == 0):
            regrets, actions = cache[ki]
            
            y = format_ra(regrets, actions)
        
        else:
            if di < dj:
                wi = 1 - (di / (di + dj))
                wj = 1 - wi

            else:
                wj = 1 - (dj / (di + dj))
                wi = 1 - wj
            
            regrets_i, actions = cache[ki]
            yi = format_ra(regrets_i, actions)
            
            regrets_j, actions_j = cache[kj]
            yj = format_ra(regrets_j, actions_j)
        
            y = wi * yi + wj * yj
            

        return y, actions
