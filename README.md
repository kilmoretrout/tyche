# Tyche

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/gui.png?raw=true)

## Overview

Tyche is software that queries a pre-computed CFR table for 6-player No-Limit Texas Holdem.  I intend to add other games (Omaha poker) and more features to the software as time goes on.

## Install

### Downloading the tables

The tables are available for purchase at:

The uncompressed tables and buckets take up roughly ~ 105 Gb of disk space at current (9.2 Gb compressed).  The file location of the tables directory should be specified in ```settings.config``` in the root of the repository.

### Conda

```
conda create -n "tyche" python=3.11
pip install -r requirements.txt
```

## Usage

After running setup.py or from the repos root:

```
import itertools

suits = "shdc"
ranks = "23456789TJQKA"

# our indexing of cards
cards = list(itertools.product(ranks, suits))
cards = [u + v for (u, v) in cards]

from tyche import TabularStrategy
strat = TabularStrategy() # will load from the default location in settings.config

# strategy for the history of four folds, street = 0 (preflop), cards = [2s, 2h]
# p: ndarray, (169, n_actions), counterfactual regrets for each action
# idx: int, bucket or index of the private state. learned strategy is p[idx]
# actions: list[str], list of actions for this node in the game tree
p, idx, actions = strat.get('ffff', 0, [0, 1])

print(actions)
# f = fold, c = check / call
# raises are in terms of the previous raise, bets in terms of the pot, and r(-1) or b(-1) means all-in
['f', 'c', 'r(1.0000)', 'r(1.5000)', 'r(2.0000)', 'r(3.0000)', 'r(4.0000)', 'r(6.0000)', 'r(8.0000)', 'r(12.0000)', 'r(-1.0)']
```
