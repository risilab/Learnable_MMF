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

# For other datasets
from data_loader import *

# Heuristics to find indices
from heuristics import *

# Learnable MMF
from learnable_mmf_model import Learnable_MMF

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Random MMF')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'kron', help = 'Dataset')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../data/', help = 'Data folder')
    parser.add_argument('--cayley_order', '-cayley_order', type = int, default = 3, help = 'Cayley order')
    parser.add_argument('--cayley_depth', '-cayley_depth', type = int, default = 4, help = 'Cayley depth')
    parser.add_argument('--L', '-L', type = int, default = 2, help = 'L')
    parser.add_argument('--K', '-K', type = int, default = 2, help = 'K')
    parser.add_argument('--drop', '-drop', type = int, default = 1, help = 'Drop rate')
    parser.add_argument('--dim', '-dim', type = int, default = 1, help = 'Dimension left at the end')
    parser.add_argument('--epochs', '-epochs', type = int, default = 128, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--visual', '-v', type = int, default = 0, help = 'Visualization or not')
    parser.add_argument('--heuristics', '-heuristics', type = str, default = 'smart', help = 'Heuristics to find indices')
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

# Constants
N = A.size(0)
K = args.K
L = args.L
drop = args.drop
dim = args.dim

# Heuristics to find indices
A_sparse = A.to_sparse()
if args.heuristics == 'random':
    wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
else:
    if drop == 1:
        wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
    else:
        wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)

# Execute the randomized MMF
model = Learnable_MMF(A, L, K, drop, dim, wavelet_indices, rest_indices)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

# Training
all_losses = []
all_errors = []
norm = torch.norm(A, p = 'fro')

best = 1e9
for epoch in range(args.epochs):
    print('Epoch', epoch, ' --------------------')
    t = time.time()
    optimizer.zero_grad()

    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

    loss = torch.norm(A - A_rec, p = 'fro')
    loss.backward()

    error = loss / norm
    all_losses.append(loss.item())
    all_errors.append(error.item())

    print('Loss =', loss.item())
    print('Error =', error.item())
    print('Time =', time.time() - t)
    LOG.write('Loss = ' + str(loss.item()) + '\n')
    LOG.write('Error = ' + str(error.item()) + '\n')
    LOG.write('Time = ' + str(time.time() - t) + '\n')

    # Early stopping
    if loss.item() > best:
        break
    best = loss.item()

    if epoch % 1000 == 0:
        torch.save(model.state_dict(), args.dir + '/' + args.name + '.model')
        print('Save model to file')

    for l in range(L):
        X = torch.Tensor(model.all_O[l].data)
        G = torch.Tensor(model.all_O[l].grad.data)
        Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
        tau = args.learning_rate
        Y = torch.matmul(torch.matmul(torch.inverse(torch.eye(K) + tau / 2 * Z), torch.eye(K) - tau / 2 * Z), X)
        model.all_O[l].data = Y.data

torch.save(model.state_dict(), args.dir + '/' + args.name + '.model')
print('Save model to file')

# Testing
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

# Plot the Frobenius norm after each rotation matrix
diff_rec = torch.abs(A - A_rec)
final_loss = torch.norm(diff_rec, p = 'fro')
final_error = final_loss / norm

print('----------------------------------')
print('Final loss =', final_loss.item())
LOG.write('Final loss =' + str(final_loss.item()) + '\n')
print('Final error =', final_error.item())
LOG.write('Final error =' + str(final_error.item()) + '\n')

all_losses.append(final_loss.item())
all_errors.append(final_error.item())

# Plot curve
def plot_curve(arr, title, xlabel, ylabel, file_name):
    plt.clf()
    plt.plot(arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)

# Draw matrix
def draw_mat(mat, file_name, title, dpi = 300):
    plt.clf()
    plt.matshow(mat)
    plt.colorbar()
    plt.title(title)
    plt.savefig(file_name, dpi = dpi)

# Draw wavelet
def draw_wavelet(wavelet, file_name, title):
    wavelet = wavelet.flatten()
    plt.clf()
    plt.plot(wavelet)
    plt.title(title)
    plt.savefig(file_name)

if args.visual == 1:
    plot_curve(all_losses, 'Frobenius Loss', 'Number of epochs', 'Loss', args.dir + '/' + args.name + '.loss_curve_1.png')
    plot_curve(all_errors, 'Normalized Frobenius Loss', 'Number of epochs', 'Loss', args.dir + '/' + args.name + '.loss_curve_2.png')

    draw_mat(A.detach().cpu().numpy(), args.dir + '/' + args.name + '.input.png', 'Input', dpi = 1200)
    draw_mat(A_rec.detach().cpu().numpy(), args.dir + '/' + args.name + '.recontruction.png', 'Reconstruction', dpi = 1200)

    draw_mat(D.detach().cpu().numpy(), args.dir + '/' + args.name + '.D.png', 'D', dpi = 1200)
    draw_mat(diff_rec.detach().cpu().numpy(), args.dir + '/' + args.name + '.difference.png', 'Frobenius norm = ' + str(final_loss.item()))

print('Done')
LOG.close()
