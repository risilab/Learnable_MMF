#!/bin/bash

program=sparse_mmf_run

# Cora citation graph
dataset=cora

#### Other options for data. Remember to change the MMF hyperparameters accordingly.
#### Kronecker product matrix
# dataset=kron

#### Cycle graph
# dataset=cycle

#### MNIST RBF kernel gram matrix
# dataset=mnist

#### Cayley tree
# dataset=cayley

#### Karate club network
# dataset=karate

#### Citation graphs
# dataset=citeseer

#### Minnesota road network
# dataset=minnesota

mkdir $program
cd $program
mkdir $dataset
cd ..
dir=$program/$dataset

learning_rate=0.1
device=cpu
epochs=100

K=16
L=250
drop=10
dim=208
heuristics=smart

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.heuristics.${heuristics}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --heuristics=$heuristics --device=$device

