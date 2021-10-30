#!/bin/bash

program=learnable_mmf_run

# Cayley tree
dataset=cayley

#### Other options for data. Remember to change the MMF hyperparameters accordingly.
#### Kronecker product matrix
# dataset=kron

#### Cycle graph
# dataset=cycle

#### MNIST RBF kernel gram matrix
# dataset=mnist

#### Karate club network
# dataset=karate

dir1=${program}
mkdir $dir1
cd $dir1
dir2=${dataset}
mkdir $dir2
cd ..
dir=${dir1}/${dir2}

learning_rate=1e-3
device=cpu
epochs=100000

K=8
L=145
drop=1
dim=16

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --device=$device

K=8
L=129
drop=1
dim=32

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --device=$device

K=8
L=113
drop=1
dim=48

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --device=$device

K=8
L=97
drop=1
dim=64

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --device=$device

K=8
L=81
drop=1
dim=80

name=${program}.dataset.${dataset}.L.${L}.K.${K}.drop.${drop}.dim.${dim}.epochs.${epochs}.learning_rate.${learning_rate}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate --device=$device
