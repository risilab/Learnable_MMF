#!/bin/bash

program=nystrom_run

# Cayley tree
dataset=cayley
cayley_order=3
cayley_depth=4

#### Other options for data. Remember to change the hyperparameters accordingly.
#### Kronecker product matrix
# dataset=kron

#### Cycle graph
# dataset=cycle

#### MNIST RBF kernel gram matrix
# dataset=mnist

#### Cayley tree
# dataset=cayley

#### Citation graphs
# dataset=cora
# dataset=citeseer
# dataset=WebKB

#### Minnesota road network
# dataset=minnesota

device=cpu

dim=4
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim

dim=8
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim

dim=16
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim

dim=32
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim

dim=64
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim

dim=128
python3 $program.py --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --dim=$dim
