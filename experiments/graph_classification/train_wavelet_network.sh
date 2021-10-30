#!/bin/bash

program=train_wavelet_network
data_folder=../../data/

# Dataset
dataset=MUTAG

#### Other datasets
# dataset=PTC
# dataset=DD
# dataset=NCI1

# Wavelet basis found by the baseline MMF
wavelet_type=baseline_mmf_basis

# Wavelet basis found by the learnable MMF
# wavelet_type=learnable_mmf_basis

mkdir ${program}
cd ${program}
mkdir ${dataset}
cd ..
dir=${program}/${dataset}

# Baseline MMF wavelet basis
adjs=${wavelet_type}/${dataset}/${dataset}.${wavelet_type}.adjs.pt
laplacians=${wavelet_type}/${dataset}/${dataset}.${wavelet_type}.laplacians.pt
mother_wavelets=${wavelet_type}/${dataset}/${dataset}.${wavelet_type}.mother_wavelets.pt
father_wavelets=${wavelet_type}/${dataset}/${dataset}.${wavelet_type}.father_wavelets.pt

# Hyper-parameters
num_epoch=256
num_layers=6
hidden_dim=32

# Cross-validation training
for split in 0 1 2 3 4 5 6 7 8 9 
do
name=${program}.dataset.${dataset}.split.${split}.num_epoch.${num_epoch}.num_layers.${num_layers}.hidden_dim.${hidden_dim}
python3 ${program}.py --dataset=$dataset --data_folder=$data_folder --dir=$dir --name=$name --num_epoch=$num_epoch --adjs=$adjs --laplacians=$laplacians --mother_wavelets=$mother_wavelets --father_wavelets=$father_wavelets --split=$split --num_layers=$num_layers --hidden_dim=$hidden_dim
done

# Summary of results
cat ${program}/${dataset}/*.log | grep 'Best accuracy:'
