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
import matplotlib.pyplot as plt

# Nystrom model
from nystrom_model import nystrom_model

# Data loader
from data_loader import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Nystrom')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../data/', help = 'Data folder')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'kron', help = 'Dataset')
    parser.add_argument('--cayley_order', '-cayley_order', type = int, default = 3, help = 'Cayley order')
    parser.add_argument('--cayley_depth', '-cayley_depth', type = int, default = 4, help = 'Cayley depth')
    parser.add_argument('--dim', '-dim', type = int, default = 8, help = 'Number of columns left')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()

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

# Data loader
dataset = args.dataset
data_folder = args.data_folder
A = None
if dataset == 'kron':
    A = kron_def()
if dataset == 'cycle':
    A = cycle_def()
if dataset == 'cora' or dataset == 'citeseer' or dataset == 'WebKB':
    adj, L_norm, features, labels, paper_ids, labels_list = citation_def(data_folder = data_folder, dataset = dataset)
    A = L_norm
if dataset == 'mnist':
    A = mnist_def(data_folder = data_folder)
if dataset == 'road':
    A = road_def(data_folder = data_folder)
if dataset == 'karate':
    A = karate_def(data_folder = data_folder)
if dataset == 'cayley':
    A, cayley_node_x, cayley_node_y, edges = cayley_def(cayley_order = args.cayley_order, cayley_depth = args.cayley_depth)

assert A is not None

# Nystrom method
A_rec, C, W_inverse = nystrom_model(A, dim = args.dim)
