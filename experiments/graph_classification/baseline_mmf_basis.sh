#!/bin/bash

program=baseline_mmf_basis
data_folder=../../data/

# Dataset
dataset=MUTAG

# Other datasets
# dataset=PTC
# dataset=DD
# dataset=NCI1

dir=${program}
mkdir ${dir}
cd ${dir}
mkdir ${dataset}
cd ..

dim=2
name=${dataset}.${program}
python3 $program.py --data_folder=$data_folder --dir=$dir --dataset=$dataset --name=$name --dim=$dim
