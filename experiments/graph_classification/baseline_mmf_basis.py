import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import time

# Baseline MMF
import sys
sys.path.append('../../source/')
from baseline_mmf_model import Baseline_MMF

# Data loader
from Dataset import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Original MMF')
    parser.add_argument('--dir', '-dir', type = str, default = 'wavelet_basis/', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = '.', help = 'Graph kernel benchmark dataset')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../../data/', help = 'Data folder')
    parser.add_argument('--dim', '-dim', type = int, default = 2, help = 'Dimension left at the end')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
log_name = args.dir + "/" + args.name + ".log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

print(args.name)
print(args.dir)

# Data loader
data_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.dat'
meta_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.meta'

data = Dataset(data_fn, meta_fn)
num_data = data.nMolecules

# Execute the baseline MMF
adjs = []
laplacians = []

mother_coeffs = []
father_coeffs = []

mother_wavelets = []
father_wavelets = []

for sample in range(num_data):
    molecule = data.molecules[sample]
    N = molecule.nAtoms
    
    if N > 512:
        print('N is too big')
        adjs.append(None)
        laplacians.append(None)
        mother_coeffs.append(None)
        father_coeffs.append(None)
        mother_wavelets.append(None)
        father_wavelets.append(None)
        continue

    # Adjacency matrix
    adj = torch.zeros(N, N)
    for v in range(N):
        neighbors = molecule.atoms[v].neighbors
        adj[v, neighbors] = 1
    adjs.append(adj)

    adj = torch.Tensor(adj) + torch.eye(N)
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Normalized graph Laplacian
    L_norm = torch.matmul(torch.matmul(DD, torch.diag(D) - adj), DD)
    laplacians.append(L_norm)
    A = L_norm

    # Factorization
    L = N - args.dim
    model = Baseline_MMF(N, L, args.dim)

    if L > 0:
        A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets_, father_wavelets_ = model(A)
    
        diag = torch.diag(mother_coefficients).unsqueeze(dim = 0)
        values, indices = torch.sort(diag, descending = True)
        mother_coeffs.append(values)

        values, indices = torch.sort(father_coefficients.flatten(), descending = True)
        father_coeffs.append(values.unsqueeze(dim = 0))

        mother_wavelets.append(mother_wavelets_.unsqueeze(dim = 0))
        father_wavelets.append(father_wavelets_.unsqueeze(dim = 0))
    else:
        print('Bad data')
        mother_coeffs.append(None)
        father_coeffs.append(None)
        mother_wavelets.append(None)
        father_wavelets.append(None)

    if (sample + 1) % 10 == 0:
        print('Done finding wavelet basis for ', sample + 1, ' molecules')

assert len(adjs) == num_data
assert len(laplacians) == num_data
assert len(mother_coeffs) == num_data
assert len(father_coeffs) == num_data
assert len(mother_wavelets) == num_data
assert len(father_wavelets) == num_data

torch.save(adjs, args.dir + '/' + args.dataset + '/' + args.name + '.adjs.pt')
torch.save(laplacians, args.dir + '/' + args.dataset + '/' + args.name + '.laplacians.pt')
torch.save(mother_coeffs, args.dir + '/' + args.dataset + '/' + args.name + '.mother_coeffs.pt')
torch.save(father_coeffs, args.dir + '/' + args.dataset + '/' + args.name + '.father_coeffs.pt')
torch.save(mother_wavelets, args.dir + '/' + args.dataset + '/' + args.name + '.mother_wavelets.pt')
torch.save(father_wavelets, args.dir + '/' + args.dataset + '/' + args.name + '.father_wavelets.pt')
print('Done')
LOG.close()
