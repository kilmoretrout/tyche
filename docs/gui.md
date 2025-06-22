# TycheGUI

## Running the GUI

After following the install instructions in [README.md](https://github.com/kilmoretrout/tyche/blob/main/README.md):

```
streamlit run src/gui.py
```

## About

Tyche includes a tool to visualize the strategy for various private hole card states and public game histories (see [methods.md](https://github.com/kilmoretrout/tyche/blob/main/docs/methods.md) for more information). 

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/gui.png?raw=true)

The user can manually enter game histories and modify the public and private cards such that he can view the learned strategy for any given history. The application also keeps track of the "reach" or posterior probability of each possible private hole card pair given their actions in the game thus far and are learned strategy:

![alt text](https://github.com/kilmoretrout/tyche/blob/main/docs/images/reach.png?raw=true)





