# Tyche

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/gui.png?raw=true)

## Overview

Tyche is software that queries a pre-computed CFR table for 6-player No-Limit Texas Holdem.  I intend to add other games (Omaha poker) and more features to the software as time goes on.

## Install

### Downloading the tables

The tables are available for purchase at (see ```docs/methods.md``` for their technical specifications):
https://www.patreon.com/c/tychepoker/shop

The uncompressed tables and buckets take up roughly ~ 105 Gb of disk space at current (9.2 Gb compressed).  The file location of the tables directory should be specified in ```tyche/settings.config``` in the root of the repository.

### Conda

```
conda create -n "tyche" python=3.11
git clone https://github.com/kilmoretrout/tyche.git
cd tyche/
python3 setup.py install
```

## Usage

### GUI

```
streamlit run gui.py
```

See ```docs/gui.md``` for more info.

### Using the Python library

Getting the strategy for a given history and public state:

```
import itertools

suits = "shdc"
ranks = "23456789TJQKA"

# our indexing of cards
cards = list(itertools.product(ranks, suits))
cards = [u + v for (u, v) in cards]

from tyche import TabularStrategy
strat = TabularStrategy() # will load from the default location in settings.config, can take a little while

# strategy for the history of four folds, street = 0 (preflop), cards = [2s, 2h]
# p: ndarray, (169, n_actions), counterfactual regrets for each action
# idx: int, bucket or index of the private state. learned strategy is p[idx]
# actions: list[str], list of actions for this node in the game tree
p, idx, actions = strat.get('ffff', 0, [0, 1]) # fast

print(actions)
# f = fold, c = check / call
# raises are in terms of the previous raise, bets in terms of the pot, and r(-1) or b(-1) means all-in
# ['f', 'c', 'r(1.0000)', 'r(1.5000)', 'r(2.0000)', 'r(3.0000)', 'r(4.0000)', 'r(6.0000)', 'r(8.0000)', 'r(12.0000)', 'r(-1.0)']

print(p[idx])
"""
[5.27311873e-04 4.95993020e-03 0.00000000e+00 0.00000000e+00
 8.39730501e-01 1.09757693e-03 7.95078352e-02 5.50571713e-04
 6.90683499e-02 9.83780832e-04 5.29770739e-04 9.63865954e-04
 2.08038301e-03]

# we can get the unnormalized regrets as well:
p, _, _ = strat.get('ffff', 0, [0, 1], norm = False)

print(p[idx])
"""
array([2.3804000e+04, 2.2390200e+05, 0.0000000e+00, 0.0000000e+00,
       3.7907256e+07, 4.9547000e+04, 3.5891560e+06, 2.4854000e+04,
       3.1178950e+06, 4.4410000e+04, 2.3915000e+04, 4.3511000e+04,
       9.3913000e+04])
"""
```
