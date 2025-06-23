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
