#!/bin/bash

program=baseline_mmf_run

# Karate club network
dataset=karate

#### Other options for data. Remember to change the MMF hyperparameters accordingly.
#### Kronecker product matrix
# dataset=kron

#### Cycle graph
# dataset=cycle

#### MNIST RBF kernel gram matrix
# dataset=mnist

#### Cayley tree
# dataset=cayley

mkdir ${program}
cd ${program}
mkdir ${dataset}
cd ..
dir=${program}/${dataset}

device=cpu

# Number of times of running to average the results
num_times=10

L=26
dim=8
name=${program}.dataset.${dataset}.L.${L}.dim.${dim}.num_times.${num_times}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --dim=$dim --num_times=$num_times --device=$device

L=22
dim=12
name=${program}.dataset.${dataset}.L.${L}.dim.${dim}.num_times.${num_times}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --dim=$dim --num_times=$num_times --device=$device

L=18
dim=16
name=${program}.dataset.${dataset}.L.${L}.dim.${dim}.num_times.${num_times}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --dim=$dim --num_times=$num_times --device=$device

L=14
dim=20
name=${program}.dataset.${dataset}.L.${L}.dim.${dim}.num_times.${num_times}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --dim=$dim --num_times=$num_times --device=$device

L=10
dim=24
name=${program}.dataset.${dataset}.L.${L}.dim.${dim}.num_times.${num_times}.device.${device}
python3 $program.py --dir=$dir --name=$name --dataset=$dataset --L=$L --dim=$dim --num_times=$num_times --device=$device

