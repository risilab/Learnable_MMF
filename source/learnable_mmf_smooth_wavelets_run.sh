#!/bin/bash

program=learnable_mmf_smooth_wavelets_run

# Cayley tree
dataset=cayley
cayley_order=2
cayley_depth=4

#### Other options for data. Remember to change the MMF hyperparameters accordingly.
#### Kronecker product matrix
# dataset=kron

#### Cycle graph
# dataset=cycle

#### MNIST RBF kernel gram matrix
# dataset=mnist

#### Cayley tree
# dataset=cayley

dir1=${program}
mkdir $dir1
cd $dir1
dir2=${dataset}_order_${cayley_order}_depth_${cayley_depth}
mkdir $dir2
cd ..
dir=${dir1}/${dir2}

learning_rate=1e-3
device=cpu
epochs=20000

K=4
L=42
drop=1
dim=4
alpha=0.5
name=${program}.dataset.${dataset}.cayley_order.${cayley_order}.cayley_depth.${cayley_depth}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.alpha.${alpha}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --cayley_order=$cayley_order --cayley_depth=$cayley_depth --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --alpha=$alpha --device=$device
