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
from mpl_toolkits.mplot3d import Axes3D
import base64
from io import BytesIO

# For other datasets
from data_loader import *

# Baseline MMF
from baseline_mmf_model import Baseline_MMF

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Baseline MMF')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../data/', help = 'Data folder')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'kron', help = 'Dataset')
    parser.add_argument('--L', '-L', type = int, default = 2, help = 'L')
    parser.add_argument('--dim', '-dim', type = int, default = 1, help = 'Dimension left at the end')
    parser.add_argument('--num_times', '-num_times', type = int, default = 10, help = 'Number of times to run for statistics')
    parser.add_argument('--cayley_order', '-cayley_order', type = int, default = 2, help = 'Cayley order')
    parser.add_argument('--cayley_depth', '-cayley_depth', type = int, default = 5, help = 'Cayley depth')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    parser.add_argument('--visual', '-visual', type = int, default = 0, help = 'Visualization or not')
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

# Dataloader
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
N = A.size(0)
L = args.L
dim = args.dim

# Fixes for the baseline MMF
K = 2
drop = 1

# Execute the baseline MMF
model = Baseline_MMF(N, L, dim)

all_losses = []
for i in range(args.num_times):
    t = time.time()
    A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model(A)
    print('Time =', time.time() - t)
    LOG.write('Time = ' + str(time.time() - t) + '\n')

    # Save the wavelets to file
    torch.save(mother_wavelets.detach(), args.dir + '/' + args.name + '.mother_wavelets.pt')
    torch.save(father_wavelets.detach(), args.dir + '/' + args.name + '.father_wavelets.pt')

    # Plot the Frobenius norm after each rotation matrix
    diff_rec = torch.abs(A - A_rec)
    loss = torch.norm(diff_rec, p = 'fro')
    norm = torch.norm(A, p = 'fro')
    error = loss / norm

    print('----------------------------------')
    print('Loss =', loss.item())
    LOG.write('Loss =' + str(loss.item()) + '\n')
    print('Error =', error.item())
    LOG.write('Error =' + str(error.item()) + '\n')
    all_losses.append(loss.item())

# Statistics
all_losses = np.array(all_losses)
mean_loss = np.mean(all_losses)
std_loss = np.std(all_losses)

print('----------------------------------')
print('Mean Loss =', mean_loss)
LOG.write('Mean Loss =' + str(mean_loss) + '\n')
print('STD Loss =', std_loss)
LOG.write('STD Loss =' + str(std_loss) + '\n')

print('Done')
LOG.close()
