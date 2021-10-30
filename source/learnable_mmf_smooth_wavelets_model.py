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

# Data loader
from data_loader import *

# Learnable MMF with smooth wavelets
class Learnable_MMF_Smooth_Wavelets(nn.Module):
    def __init__(self, A, L, K, drop, dim, device = 'cpu'):
        super(Learnable_MMF_Smooth_Wavelets, self).__init__()
        
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
        self.dim = dim
        assert self.dim == self.N - self.L * self.drop

        # Device
        self.device = device

        # +-----------+
        # | Neighbors |
        # +-----------+

        # Non-zero elements
        edges = torch.nonzero(A).detach().cpu().numpy()
        neighbors = []
        for node in range(self.N):
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
        for v1 in range(self.N):
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
        for v1 in range(self.N):
            neighbors[v1] += two_hops[v1]

        # Randomization of the indices
        self.selected_indices = []
        self.active_indices = []
        self.drop_indices = []

        active_index = torch.ones(self.N)

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

            selected_index = torch.zeros(self.N)
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
            U = torch.eye(self.N).to(device = self.device)
            U[outer == 1] = O.flatten()

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))

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
            O = self.all_O[l].to(device = self.device)

            # Full Jacobian rotation matrix
            U = torch.eye(self.N).to(device = self.device)
            U[outer == 1] = O.flatten()

            if l == 0:
                right = U
            else:
                right = torch.matmul(U, right)

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))
            
            # Wavelet
            wavelet = torch.reshape(A[self.drop_indices[l] == 1], (self.drop, self.N))
            wavelets.append(wavelet)

        # Diagonal left
        active_index = self.active_indices[self.L].to(device = self.device)
        left_index = torch.outer(active_index, active_index)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index
        
        # Core wavelets
        core_wavelets = []
        for n in range(self.N):
            if self.active_indices[self.L][n] == 1:
                ind = torch.zeros(self.N)
                ind[n] = 1
                wavelet = torch.reshape(A[ind == 1], (1, self.N))
                core_wavelets.append(wavelet)

        # Mother coefficients
        outer = torch.outer(1 - active_index, 1 - active_index)
        mother_coefficients = torch.reshape(D[outer == 1], (self.N - self.dim, self.N - self.dim))

        # Father coefficients
        outer = torch.outer(active_index, active_index)
        father_coefficients = torch.reshape(D[outer == 1], (self.dim, self.dim))

        # Mother wavelets
        mother_wavelets = torch.cat(wavelets, dim = 0)

        # Father wavelets
        father_wavelets = torch.cat(core_wavelets, dim = 0)

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        # Result
        return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets


# +------------------------+
# | Learnable MMF training |
# +------------------------+

def learnable_mmf_smooth_wavelets_train(A, L, K, drop, dim, epochs = 10000, learning_rate = 1e-3, Lambda = 1, alpha = 0.5, early_stop = True):
    model = Learnable_MMF_Smooth_Wavelets(A, L, K, drop, dim)
    optimizer = Adagrad(model.parameters(), lr = learning_rate)

    # Training
    best = 1e9
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()

        A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

        # Reconstruction loss
        rec_loss = torch.norm(A - A_rec, p = 'fro')

        # Smooth loss
        smooth_loss = torch.pow(torch.sum(torch.diag(torch.matmul(mother_wavelets, torch.matmul(A, torch.transpose(mother_wavelets, 0, 1))))), alpha)
        smooth_loss += torch.pow(torch.sum(torch.diag(torch.matmul(father_wavelets, torch.matmul(A, torch.transpose(father_wavelets, 0, 1))))), alpha)

        # Total loss
        loss = rec_loss + Lambda * smooth_loss
        loss.backward()

        if epoch % 1000 == 0:
            print('---- Epoch', epoch, '----')
            print('Reconstruction loss =', rec_loss.item())
            print('Smoothing loss =', smooth_loss.item())
            print('Loss =', loss.item())
            print('Time =', time.time() - t)

        if loss.item() < best:
            best = loss.item()
        else:
            if early_stop:
                print('Early stop at epoch', epoch)
                break

        # Update parameter
        if epoch + 1 < epochs:
            for l in range(L):
                X = torch.Tensor(model.all_O[l].data)
                G = torch.Tensor(model.all_O[l].grad.data)
                Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
                tau = learning_rate
                Y = torch.matmul(torch.matmul(torch.inverse(torch.eye(K) + tau / 2 * Z), torch.eye(K) - tau / 2 * Z), X)
                model.all_O[l].data = Y.data

    # Return the result
    A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

    # Reconstruction loss
    rec_loss = torch.norm(A - A_rec, p = 'fro')

    # Smooth loss
    smooth_loss = torch.pow(torch.sum(torch.diag(torch.matmul(mother_wavelets, torch.matmul(A, torch.transpose(mother_wavelets, 0, 1))))), alpha)
    smooth_loss += torch.pow(torch.sum(torch.diag(torch.matmul(father_wavelets, torch.matmul(A, torch.transpose(father_wavelets, 0, 1))))), alpha)

    # Total loss
    loss = rec_loss + Lambda * smooth_loss

    print('---- Final loss ----')
    print('Reconstruction loss =', rec_loss.item())
    print('Smooth loss =', smooth_loss.item())
    print('Loss =', loss.item())

    return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets

