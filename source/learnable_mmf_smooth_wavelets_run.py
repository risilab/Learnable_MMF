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

from data_loader import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Random MMF')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'Kronecker', help = 'Dataset')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../data/', help = 'Data folder')
    parser.add_argument('--L', '-L', type = int, default = 2, help = 'L')
    parser.add_argument('--K', '-K', type = int, default = 2, help = 'K')
    parser.add_argument('--drop', '-drop', type = int, default = 1, help = 'Drop rate')
    parser.add_argument('--dim', '-dim', type = int, default = 1, help = 'Dimension left at the end')
    parser.add_argument('--epochs', '-epochs', type = int, default = 128, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('--Lambda', '-Lambda', type = float, default = 1, help = 'Wavelet smoothening coefficient (Lambda)')
    parser.add_argument('--alpha', '-alpha', type = float, default = 0.5, help = 'Wavelet smoothening coefficient (alpha)')
    parser.add_argument('--cayley_order', '-cayley_order', type = int, default = 2, help = 'Cayley order')
    parser.add_argument('--cayley_depth', '-cayley_depth', type = int, default = 5, help = 'Cayley depth')
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

# Random MMF
class Random_MMF(nn.Module):
    def __init__(self, A, L, K, drop, dim, neighbors, device = 'cpu'):
        super(Random_MMF, self).__init__()
        
        # Matrix
        self.A = A

        # Size of the matrix
        self.N = A.size(0)
        assert A.dim() == 2
        assert A.size(1) == self.N

        # Number of resolutions
        self.L = L

        # Size of the Jacobian rotation matrix
        self.K = K

        # Number of rows/columns to drop in each iteration
        self.drop = drop

        # Number of left rows/columns at the end
        self.left = self.N - self.L * self.drop
        assert self.left == dim

        # Device
        self.device = device

        # Randomization of the indices
        self.selected_indices = []
        self.active_indices = []
        self.drop_indices = []

        active_index = torch.ones(N)

        for l in range(self.L):
            self.active_indices.append(torch.Tensor(active_index.data).to(device = self.device))

            left_list = np.array([i for i in range(self.N) if active_index[i] == 1])
            perm = torch.randperm(left_list.shape[0]).detach().cpu().numpy()
            assert left_list.shape[0] == perm.shape[0]

            # Select the main vertex/node
            main_node = left_list[perm[0]]
            index = [main_node]
            if len(neighbors[main_node]) == self.K - 1:
                index = index + neighbors[main_node]
            else:
                if len(neighbors[main_node]) < self.K - 1:
                    index = index + neighbors[main_node]
                    for i in range(self.K - 1 - len(neighbors[main_node])):
                        while True:
                            v = left_list[np.random.randint(left_list.shape[0])]
                            if v in index:
                                continue
                            index.append(v)
                            break
                else:
                    perm = torch.randperm(len(neighbors[main_node])).detach().cpu().numpy()
                    index = index + [neighbors[main_node][perm[i]] for i in range(self.K - 1)]

            assert len(index) == self.K
            index = np.array(index)

            selected_index = torch.zeros(N)
            for k in range(self.K):
                selected_index[index[k]] = 1
            self.selected_indices.append(selected_index.to(device = self.device))

            drop_index = torch.zeros(self.N)
            for i in range(self.drop):
                ind = index[i]
                active_index[ind] = 0
                drop_index[ind] = 1
            self.drop_indices.append(drop_index.to(device = self.device))
        
        self.active_indices.append(torch.Tensor(active_index.data).to(device = self.device))

        # Initialization of the Jacobi rotation matrix
        self.all_O = torch.nn.ParameterList()
        
        # The current matrix
        A = torch.Tensor(self.A.data)

        # Wavelets
        # self.wavelets = []

        for l in range(self.L):
            # Randomization of the indices
            index = self.selected_indices[l]
            
            # Outer product map
            outer = torch.outer(index, index)

            # Eigen-decomposition
            A_part = torch.matmul(A[index == 1], torch.transpose(A[index == 1], 0, 1))
            values, vectors = torch.eig(torch.reshape(A_part, (self.K, self.K)), True)

            # Rotation matrix
            O = torch.nn.Parameter(vectors.transpose(0, 1).data, requires_grad = True)
            self.all_O.append(O)

            # Full Jacobian rotation matrix
            U = torch.eye(N).to(device = self.device)
            U[outer == 1] = O.flatten()

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))

            # Wavelet
            # wavelet = torch.reshape(A[self.drop_indices[l] == 1], (self.drop, self.N))
            # self.wavelets.append(wavelet)

    def forward(self):
        # The current matrix
        A = self.A

        # Wavelets
        wavelets = []

        # For each resolution
        for l in range(self.L):
            # Randomization of the indices
            index = self.selected_indices[l]
            
            # Outer product map
            outer = torch.outer(index, index)

            # Jacobi rotation matrix
            O = self.all_O[l].to(device = device)

            # Full Jacobian rotation matrix
            U = torch.eye(N).to(device = self.device)
            U[outer == 1] = O.flatten()

            if l == 0:
                right = U
            else:
                right = torch.matmul(U, right)

            # Wavelet
            # wavelet = torch.reshape(A[self.drop_indices[l] == 1], (self.drop, self.N))
            # wavelets.append(wavelet)

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))
            
            # Wavelet
            wavelet = torch.reshape(A[self.drop_indices[l] == 1], (self.drop, self.N))
            wavelets.append(wavelet)

        # Diagonal left
        left_index = self.active_indices[self.L].to(device = self.device)
        left_index = torch.outer(left_index, left_index)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index
        
        # Core wavelets
        core_wavelets = []
        for n in range(self.N):
            if self.active_indices[self.L][n] == 1:
                ind = torch.zeros(self.N)
                ind[n] = 1
                wavelet = torch.reshape(A[ind == 1], (1, N))
                core_wavelets.append(wavelet)

        # Mother wavelets 2
        active_index = self.active_indices[self.L].to(device = self.device)
        mother_wavelets_2 = torch.reshape(right[active_index == 0], (self.N - self.left, self.N))

        # Father wavelets 2
        father_wavelets_2 = torch.reshape(right[active_index == 1], (self.left, self.N))

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        # Result
        return A_rec, D, wavelets, core_wavelets, mother_wavelets_2, father_wavelets_2

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

# Non-zero elements
edges = torch.nonzero(A).detach().cpu().numpy()
neighbors = []
for node in range(N):
    neighbors.append([])

# One-hop neighbors
for edge_index in range(edges.shape[0]):
    v1 = edges[edge_index, 0]
    v2 = edges[edge_index, 1]
    if v1 == v2:
        continue
    neighbors[v1].append(v2)

# Two-hop neighbors
two_hops = []
for v1 in range(N):
    two_hops.append([])
    for v2 in neighbors[v1]:
        for v3 in neighbors[v2]:
            if v1 == v3:
                continue
            if v3 in neighbors[v1]:
                continue
            if v3 in two_hops[v1]:
                continue
            two_hops[v1].append(v3)
for v1 in range(N):
    neighbors[v1] += two_hops[v1]

# Execute the randomized MMF
model = Random_MMF(A, L, K, drop, dim, neighbors)
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

    A_rec, D, wavelets, core_wavelets, mother_wavelets_2, father_wavelets_2 = model()

    rec_loss = torch.norm(A - A_rec, p = 'fro')

    # Wavelet smoothening loss
    assert L == len(wavelets)
    for l in range(L):
        f = wavelets[l]

        term = torch.pow(torch.sum(torch.matmul(f, torch.matmul(A, torch.transpose(f, 0, 1)))), args.alpha)
        if l == 0:
            smooth_loss = term
        else:
            smooth_loss += term

    # Core-wavelet smoothening loss
    for l in range(len(core_wavelets)):
        f = core_wavelets[l]
        term = torch.pow(torch.sum(torch.matmul(f, torch.matmul(A, torch.transpose(f, 0, 1)))), args.alpha)
        smooth_loss += term

    # More on smooth loss
    smooth_loss += torch.pow(torch.sum(torch.diag(torch.matmul(mother_wavelets_2, torch.matmul(A, torch.transpose(mother_wavelets_2, 0, 1))))), args.alpha)
    smooth_loss += torch.pow(torch.sum(torch.diag(torch.matmul(father_wavelets_2, torch.matmul(A, torch.transpose(father_wavelets_2, 0, 1))))), args.alpha)

    loss = rec_loss + args.Lambda * smooth_loss
    loss.backward()

    error = rec_loss / norm
    all_losses.append(loss.item())
    all_errors.append(error.item())

    print('Recon Loss =', rec_loss.item())
    print('Smooth Loss =', smooth_loss.item())
    print('Total Loss =', loss.item())
    print('Error =', error.item())
    print('Time =', time.time() - t)
    LOG.write('Recon Loss = ' + str(rec_loss.item()) + '\n')
    LOG.write('Smooth Loss = ' + str(smooth_loss.item()) + '\n')
    LOG.write('Total Loss = ' + str(loss.item()) + '\n')
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
A_rec, D, wavelets, core_wavelets, mother_wavelets_2, father_wavelets_2 = model()

print(A)
print(A_rec)

# Plot the Frobenius norm after each rotation matrix
diff_rec = torch.abs(A - A_rec)
final_loss = torch.norm(diff_rec, p = 'fro')
final_error = final_loss / norm

print('----------------------------------')
print('Final loss =', final_loss.item())
LOG.write('Final loss =' + str(final_loss.item()) + '\n')
print('Final error =', final_error.item())
LOG.write('Final error =' + str(final_error.item()) + '\n')

# all_losses.append(final_loss.item())
all_errors.append(final_error.item())

# Plot curve
def plot_curve(arr, title, xlabel, ylabel, file_name):
    plt.clf()
    plt.plot(arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)

plot_curve(all_losses, 'Total Loss', 'Number of epochs', 'Loss', args.dir + '/' + args.name + '.total_loss.png')
plot_curve(all_errors, 'Normalized Frobenius Loss', 'Number of epochs', 'Loss', args.dir + '/' + args.name + '.recon_loss.png')

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

# Draw graph with wavelet
def draw_graph(A, wavelet, file_name):
    N = A.shape[0]
    R = 10
    nodes_x = []
    nodes_y = []
    nodes_z = []
    for k in range(N):
        alpha = 2 * np.pi * k / N
        x = R * np.cos(alpha)
        y = R * np.sin(alpha)
        z = 0
        nodes_x.append(x)
        nodes_y.append(y)
        nodes_z.append(z)
    
    plt.clf()
    fig = plt.figure(figsize = (R, R))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(nodes_x, nodes_y, nodes_z)

    for k1 in range(N):
        x1 = nodes_x[k1]
        y1 = nodes_y[k1]
        z1 = nodes_z[k1]
        for k2 in range(N):
            if A[k1, k2] == 0:
                continue
            x2 = nodes_x[k2]
            y2 = nodes_y[k2]
            z2 = nodes_z[k2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'b')

    for k in range(N):
        w = np.sum(wavelet[:, k])
        x1 = nodes_x[k]
        y1 = nodes_y[k]
        z1 = nodes_z[k]
        x2 = x1
        y2 = y1
        z2 = z1 + w
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'r')

    # PNG
    plt.savefig(file_name + '.pdf')

    # HTML
    '''
    temp = BytesIO()
    fig.savefig(temp, format = "png")
    fig_encode_bs64 = base64.b64encode(temp.getvalue()).decode('utf-8')

    html_string = """
    <h2>Graph & wavelet</h2>
    <img src = 'data:image/png;base64,{}'/>
    """.format(fig_encode_bs64)

    with open(file_name + '.html', "w") as f:
        f.write(html_string)
    '''

    # Close figures
    plt.close(fig)

# Draw Cayley graph with wavelets
def draw_cayley(A, wavelet, nodes_x, nodes_y, edges, file_name):
    N = A.shape[0]
    R = 10
    nodes_z = []
    for k in range(N):
        nodes_z.append(0)

    plt.clf()
    fig = plt.figure(figsize = (R, R))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(nodes_x, nodes_y, nodes_z)

    for edge in edges:
        u = edge[0]
        v = edge[1]
        x1 = nodes_x[u]
        y1 = nodes_y[u]
        z1 = nodes_z[u]
        x2 = nodes_x[v]
        y2 = nodes_y[v]
        z2 = nodes_z[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'b')

    for k in range(N):
        w = np.sum(wavelet[:, k])
        x1 = nodes_x[k]
        y1 = nodes_y[k]
        z1 = nodes_z[k]
        x2 = x1
        y2 = y1
        z2 = z1 + w
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'r')

    # PNG
    plt.savefig(file_name + '.pdf')
    
    # HTML
    '''
    temp = BytesIO()
    fig.savefig(temp, format = "png")
    fig_encode_bs64 = base64.b64encode(temp.getvalue()).decode('utf-8')

    html_string = """
    <h2>Graph & wavelet</h2>
    <img src = 'data:image/png;base64,{}'/>
    """.format(fig_encode_bs64)

    with open(file_name + '.html', "w") as f:
        f.write(html_string)
    '''

    # Close figures
    plt.close(fig)

draw_mat(A.detach().cpu().numpy(), args.dir + '/' + args.name + '.input.png', 'Input', dpi = 1200)
draw_mat(A_rec.detach().cpu().numpy(), args.dir + '/' + args.name + '.recontruction.png', 'Reconstruction', dpi = 1200)

draw_mat(D.detach().cpu().numpy(), args.dir + '/' + args.name + '.D.png', 'D', dpi = 1200)
draw_mat(diff_rec.detach().cpu().numpy(), args.dir + '/' + args.name + '.difference.png', 'Frobenius norm = ' + str(final_loss.item()))

for idx in range(len(wavelets)):
    if args.dataset == 'cayley':
        draw_cayley(A.detach().cpu().numpy(), wavelets[idx].detach().cpu().numpy(), 
                cayley_node_x, cayley_node_y, edges,
                args.dir + '/' + args.name + '.cayley.wavelet.' + str(idx))
    else:
        draw_graph(A.detach().cpu().numpy(), wavelets[idx].detach().cpu().numpy(), args.dir + '/' + args.name + '.wavelet.' + str(idx))
    print('Draw mother wavelet', idx)

for idx in range(len(core_wavelets)):
    if args.dataset == 'cayley':
        draw_cayley(A.detach().cpu().numpy(), core_wavelets[idx].detach().cpu().numpy(),
                cayley_node_x, cayley_node_y, edges,
                args.dir + '/' + args.name + '.cayley.core_wavelet.' + str(idx))
    else:
        draw_graph(A.detach().cpu().numpy(), core_wavelets[idx].detach().cpu().numpy(), args.dir + '/' + args.name + '.core_wavelet.' + str(idx))
    print('Draw father wavelet', idx)

for idx in range(len(wavelets)):
    wavelet = mother_wavelets_2[idx].unsqueeze(dim = 0).detach().cpu().numpy()
    if args.dataset == 'cayley':
        draw_cayley(A.detach().cpu().numpy(), wavelet,
                cayley_node_x, cayley_node_y, edges,
                args.dir + '/' + args.name + '.cayley.mother_wavelets_2.' + str(idx))
    else:
        draw_graph(A.detach().cpu().numpy(), wavelet, args.dir + '/' + args.name + '.mother_wavelets_2.' + str(idx))
    print('Draw mother wavelet 2', idx)

for idx in range(len(core_wavelets)):
    wavelet = father_wavelets_2[idx].unsqueeze(dim = 0).detach().cpu().numpy()
    if args.dataset == 'cayley':
        draw_cayley(A.detach().cpu().numpy(), wavelet,
                cayley_node_x, cayley_node_y, edges,
                args.dir + '/' + args.name + '.cayley.father_wavelets_2.' + str(idx))
    else:
        draw_graph(A.detach().cpu().numpy(), wavelet, args.dir + '/' + args.name + '.father_wavelets_2.' + str(idx))
    print('Draw father wavelet 2', idx)

print('Done')
LOG.close()
