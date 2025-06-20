# -*- coding: utf-8 -*-

import itertools
import numpy as np
from copy import deepcopy, copy

numbers = '-.0123456789'
suits = "shdc"
ranks = "23456789TJQKA"

# our indexing of cards
cards = list(itertools.product(ranks, suits))
cards = [u + v for (u, v) in cards]

try:
    from pokerkit import (
        Automation,
        BettingStructure,
        Deck,
        Card,
        HoleDealing,
        KuhnPokerHand,
        Opening,
        Operation,
        State,
        Street,
        NoLimitTexasHoldem,
        Mode
    )
except ImportError:
    raise ImportError('PokerKit not installed. Please install through pip.')
    
def measure_length(hist):
    ret = 0
    ii = 0
    
    while ii < len(hist):
        _ = np.zeros(3)
        if hist[ii] == 'f':
            _[0] = 1.
        elif hist[ii] == 'c':
            _[1] = 1.
        elif hist[ii] in ('b', 'r'):
            a = hist[ii]
            
            ii += 2
    
            nums = ''
            while hist[ii] in numbers:
                nums += hist[ii]
    
                ii += 1

        ii += 1
        
        ret += 1
        
    return ret

def history_to_vec(hist, padding = -1, noise = 0.0):
    ii = 0
    
    hist = hist.replace('|', '')
    
    sim = NLSimulator()
    street = 0
    
    ret = []
    while ii < len(hist):
        _ = np.zeros(3)
        if hist[ii] == 'f':
            _[0] = 1.
            sim.parse_action('f')
        elif hist[ii] == 'c':
            _[1] = 1.
            sim.parse_action('c')
        elif hist[ii] in ('b', 'r'):
            a = hist[ii]
            
            
            ii += 2

            nums = ''
            while hist[ii] in numbers:
                nums += hist[ii]

                ii += 1

            f = float(nums)
            
            if f > 0:
                if a == 'b':
                    action = 'b({0:04f})'.format(f)
                    
                else:
                    action = 'r({0:04f})'.format(f)
                
                sim.parse_action(action)
            else:
                stack = sim.state.stacks[sim.state.actor_index]
                f = sim.get_all_in_factor()
                
                sim.state.complete_bet_or_raise_to(stack + sim.state.bets[sim.state.actor_index])
                
            if noise != 0.:
                # perturb the value
                e = f * np.random.uniform(-noise, noise)
            else:
                e = 0.
            
            _[2] = np.log(f + e)
            
        if sim.state.street_index != street:
            sim.state.burn_card('??')
            if sim.state.street_index == 1:    
                sim.state.deal_board('??????')
            else:
                sim.state.deal_board('??')
            
            street = deepcopy(sim.state.street_index)

        ii += 1
        
        ret.append(_)
            
    history = np.array(ret)
    
    l = history.shape[0]
    if padding > 0:
        history = np.pad(history, ((0, padding - history.shape[0]), (0, 0)))
    
    return history, l

def board_to_vec(board):
    hole_vec = np.zeros(52)
    
    hole_vec[cards.index(board[0])] = 1.
    hole_vec[cards.index(board[1])] = 1.        
    
    flop_vec = np.zeros(52)
    turn_vec = np.zeros(52)
    river_vec = np.zeros(52)
    
    board_vec = np.zeros(52)
    if len(board) >= 5:
        for k in range(2, 5):
            flop_vec[cards.index(board[k])] = 1.
    
    if len(board) >= 6:
        turn_vec[cards.index(board[5])] = 1.

    if len(board) == 7:
        river_vec[cards.index(board[6])] = 1.

    card_vec = np.concatenate([hole_vec, flop_vec, turn_vec, river_vec])
        
    return card_vec

def state_to_vec(hist, board):
    h, l = history_to_vec(hist)
    
    card_vec = board_to_vec(board)
    
    return np.concatenate([h.flatten(), card_vec])

# convert from factor to pokerkit amount
def bet_amounts(state, sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 12.0]):
    amts = []
    
    bet_sizes = sizes
    
    raise_sizes = [u for u in sizes if u >= 1.0]
        
    min_bet = 8

    # use the pot
    if sum(state.bets) == 0:
        amts = [sum(np.abs(state.payoffs)) * u for u in bet_sizes]
        
        amts = [u for u in amts if u >= min_bet and u <= state.stacks[state.actor_index]]
    else: # use the previous raise
        amt = max(state.bets)
        
        if amt == 8 and state.street_index == 0:
            prev_raise = 4
        else:
            _ = np.array(deepcopy(state.bets))
            _ = _[_ != amt]
            
            if len(_) > 0:
                if max(_) == 0:
                    prev_raise = amt
                else:
                    prev_raise = amt - max(_)
            else:
                prev_raise = 4
                    
        amts = [amt + prev_raise * u for u in raise_sizes]
        amts = [u for u in amts if u <= state.stacks[state.actor_index] + state.bets[state.actor_index]]
    
    return amts

class NLSimulator(object):
    def __init__(self, n_players = 6):
        self.state = create_nolimit(n_players = n_players)
        self.history = ''
        self.round = 0

    def parse_action(self, action):
        if action == 'f':
            self.state.fold()
        elif action == 'c':
            self.state.check_or_call()
        elif (('r' in action) or ('b' in action)):
            f = action.replace('r', '').replace('b', '').replace('(', '').replace(')', '')
            f = float(f)
            
            amt = bet_amounts(self.state, sizes = [f])[0]
            
            self.state.complete_bet_or_raise_to(amt)
                
        if self.state.street_index != self.round:
            self.round = deepcopy(self.state.street_index)
        
        self.history += action
        
    def get_all_in_factor(self):
        # a bet
        if sum(self.state.bets) == 0:
            pot = sum(np.abs(self.state.payoffs))
            
            return self.state.stacks[self.state.actor_index] / pot

        else:
            amt = max(self.state.bets)    
        
            if amt == 8 and self.state.street_index == 0:
                prev_raise = 4
            else:
                _ = np.array(deepcopy(self.state.bets))
                _ = _[_ != amt]
                
                if len(_) > 0:
                    if max(_) == 0:
                        prev_raise = amt
                    else:
                        prev_raise = amt - max(_)
                else:
                    prev_raise = 4
                
            return self.state.stacks[self.state.actor_index] / prev_raise
        
class FlowGame(NLSimulator):
    def __init__(self, n_players = 2):
        super().__init__(n_players = n_players)
        
def create_nolimit(n_players = 2):
    state = NoLimitTexasHoldem.create_state(
        # Automations
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        True,  # Uniform antes?
        0,  # Antes
        (4, 8),  # Blinds or straddles
        4,  # Min-bet
        tuple([800 for u in range(n_players)]),  # Starting stacks
        n_players,  # Number of players
        mode = Mode.CASH_GAME
    )
    
    for k in range(n_players):
        state.deal_hole("????")
        
    return state
    
if __name__ == '__main__':
    sim = NLSimulator()
    
    history = 'ffcr(3.0)r(-1.0)'
    
    v = history_to_vec(history)
    
    print(v)
    
    