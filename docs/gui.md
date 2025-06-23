# TycheGUI

Tyche includes a tool to visualize the strategy for various private hole card states and public game histories (see [methods.md](https://github.com/kilmoretrout/tyche/blob/main/docs/methods.md) for more information). 

## Running the GUI

After following the install instructions in [README.md](https://github.com/kilmoretrout/tyche/blob/main/README.md):

```
streamlit run src/gui.py
```

## Strategy for acting player and history

The user can manually enter game histories and modify the public and private cards such that he can view the learned strategy for any given history. 


![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/gui.png?raw=true)

## Reach

The application also keeps track of the "reach" or posterior probability of each possible private hole card pair given their actions in the game thus far and our learned strategy:

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/reach.png?raw=true)

## Strategy over different hole card pairs

One can also visualize the probability of a particular action in the tree over each possible hole card pair (excluding the current public or board cards):

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/strat_heatmap.png?raw=true)



