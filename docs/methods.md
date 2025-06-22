# Methods

Tyche is a package for quickly querying a 6-player No-Limit Texas Holdem strategy which is meant to closely approximate an equilibrium or strategy where no individual player (position at the table) can gain an advantage by deviating from it.  The strategy was obtained through a method called Counterfactual Regret minimization or CFR (4).  Since no-limit holdem is so large i.e. it has so many different possible game histories, we first make an abstraction of the game and the different private states that each player could have.  This means allowing only certain bet and raise sizes (relative to the pot or the previous bet in the case of raises) and grouping different private states (the hole cards and public board cards) that are strategically similar into clusters or what are commonly called "buckets".  These are called the action abstraction and card abstraction respectively.  This limits the size of the game tree, which contains all possible histories for the game, such that it can fit into RAM.  

A game tree contains action nodes and terminal nodes.  Action nodes represent states of the game where a player has to make a decision and terminal nodes represent states where the game is over.  At each action node, a strategy for each private state (bucket) is stored. The CFR algorithm walks the game tree many millions of times and for each action node / private state, it updates the strategy such that it approaches a Nash equilibrium between all the players, or all of the players losses / gains are minimized.  To learn more about the technical details of the CFR algorithm and how it accomplishes this see (4). 

## Card and action abstraction

In each betting round we define some buckets that each contain some number of strategically unique sets of cards (hole + board).  In the preflop betting round there are only 169 strategically unique scenarios corresponding to groups of hole card pairs so we can consider all of them i.e. there is one bucket for each.  However, in the flop there are 1,286,792, in the turn there are 55,190,538, and in the river or final betting stage there are 2,428,287,420 (3).  Thus in the flop and onwards we abstract the cards by making some number of groups.  

For the river, for each of the possible sets we compute its rank.  We order the other possible 990 hands by their rank and create a vector where the entry is one if the hand wins against that hand, 0.5 if it ties, and 0 if it loses.  We computed a k-means clustering pf these vectors with a target of 2048 clusters, which resulted in 2012 unique cluster means.

For the flop and turn, we used the method from (2) to compute the buckets.  The method involves working recursively, backwards from the river starting with the turn and then computing the buckets for the flop.  We consider each possible unique set for the turn and compute the histogram for which clusters in the river a player will end up in when the next card is drawn.  After computing this, for each set, we perform dimensionality reduction via landmark multi-dimensional scaling (MDS) (1).  Created in part to work with large datasets, it takes a sample of the points (the landmarks), computes an embedding via MDS, and then embeds the rest of the points via their distance from the landmark points.  After reducing the dimension of the points, we again compute a k-means clustering.  This process is repeated for the flop.  For both the turn and flop the number of clusters or buckets chosen was 1024.  

The following tables show the bet and raise sizes allowed for each round as well as the number of raises allowed in each round and the total number of raises allowed in a game.  After solving the full 6 player game with a coarse action abstraction we resolved games where some number of players folded in preflop at the start of the round with a finer abstraction (more possible raise and bet sizes, and more raises allowed, etc.).  Note that in all the action abstractions, the acting player may always go all in.  In our abstracted game, all players start the round with 100 big blinds.

### Full 6-player game
| Round  | Bets (pot) | Raises (previous raise / bet) | Max Raises |
|----------|-------|--------|--------|
| Preflop   | ---  | 1.0, 2.0     |   2     |
| Flop      | 1.0, 2.0, 4.0  | 1.0, 2.0, 4.0    |    1    | 
| Turn      | 1.0, 2.0, 4.0   | 1.0, 2.0, 4.0     |    1    |
| River     | 0.5, 1.0, 2.0, 4.0  | ---     |     0   |

Reentrant round: preflop

Max total raises: 4

### One fold, 5 players
| Round  | Bets (pot) | Raises (previous raise / bet) | Max Raises |
|----------|-------|--------|--------|
| Preflop   | ---  | 1.0, 2.0, 4.0, 8.0, 12.0     |   2     |
| Flop      | 0.5, 1.0, 2.0, 4.0, 8.0, 12.0  | 1.0, 2.0, 4.0, 8.0, 12.0     |    1    | 
| Turn      | 0.5, 1.0, 2.0, 4.0, 8.0, 12.0  | 1.0, 2.0, 4.0, 8.0, 12.0     |    1    |
| River     | 0.5, 1.0, 2.0, 4.0, 8.0, 12.0  | 1.0, 2.0, 4.0, 8.0, 12.0     |     1   |

Reentrant round: preflop
Max total raises: 5

### Two folds, 4 players
| Round  | Bets (pot) | Raises (previous raise / bet) | Max Raises |
|----------|-------|--------|--------|
| Preflop   | ---  | 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0  |   2     |
| Flop      | 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0     |    2    | 
| Turn      | 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0     |    2    |
| River     | 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0    |     1   |

Reentrant round: preflop

Max total raises: 5

### Four folds, 2 players
| Round  | Bets (pot) | Raises (previous bet) | Max Raises |
|----------|-------|--------|--------|
| Preflop   | ---  | 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0     |   3    |
| Flop      | 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0     |    2    | 
| Turn      | 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0     |    2    |
| River     | 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0  | 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0    |     1   |

Reentrant round: flop

Max total raises: 6

## Solving

The coarse strategy and each resolve was computed using Monte-Carlo CFR+ (MCCFR+) with external sampling and chance sampling.  The full 6-player coarse strategy was computed with 1 billion iterations comprised of an update for each player.  Each resolve (for 2 - 5 players) was computed with 300 million iterations using the reach and regrets computed from the 6-player strategy again using the MCCFR+ algorithm.  The first iteration of Tyche (v1.0), with the finer resolves, thus included 1.9 billion CFR iterations.   

## Citations

**(1)** De Silva, Vin, and Joshua B. Tenenbaum. Sparse multidimensional scaling using landmark points. Vol. 120. technical report, Stanford University, 2004.

**(2)** Ganzfried, Sam, and Tuomas Sandholm. "Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 28. No. 1. 2014.

**(3)** Waugh, K.. “A Fast and Optimal Hand Isomorphism Algorithm.” AAAI Conference on Artificial Intelligence (2013).

**(4)** Zinkevich, Martin, et al. "Regret minimization in games with incomplete information." Advances in neural information processing systems 20 (2007).
