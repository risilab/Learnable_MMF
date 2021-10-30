#!/bin/bash

program=learnable_mmf_basis
data_folder=../../data/

# Dataset
dataset=MUTAG

# Other datasets. Remember to change the MMF hyperparameters accordingly to the size of each dataset.
# dataset=PTC
# dataset=DD
# dataset=NCI1

dir=${program}
mkdir ${dir}
cd ${dir}
mkdir ${dataset}
cd ..

K=8
drop=1
dim=8
epochs=1024
learning_rate=1e-3

name=${dataset}.${program}
python3 $program.py --data_folder=$data_folder --dir=$dir --dataset=$dataset --name=$name --K=$K --drop=$drop --dim=$dim --epochs=$epochs --learning_rate=$learning_rate
