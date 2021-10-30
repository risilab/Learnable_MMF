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

# +---------------------+
# | Learnable MMF model |
# +---------------------+

class Learnable_MMF(nn.Module):
    def __init__(self, A, L, K, drop, dim, wavelet_indices, rest_indices, device = 'cpu'):
        super(Learnable_MMF, self).__init__()
        
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

        # Given indices
        self.wavelet_indices = wavelet_indices
        self.rest_indices = rest_indices

        active_index = torch.ones(self.N)
        self.selected_indices = []

        # Initialization of the Jacobi rotation matrix
        self.all_O = torch.nn.ParameterList()
        
        # The current matrix
        A = torch.Tensor(self.A.data)

        for l in range(self.L):
            # Set the indices for this rotation
            indices = self.wavelet_indices[l] + self.rest_indices[l]
            indices.sort()
            assert len(indices) == self.K
            index = torch.zeros(self.N)
            for k in range(self.K):
                index[indices[k]] = 1
            self.selected_indices.append(index)

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

            if l == 0:
                right = U
            else:
                right = torch.matmul(U, right)

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))

            # Drop the wavelet
            active_index[self.wavelet_indices[l]] = 0

        self.final_active_index = active_index

        # Block diagonal left
        left_index = torch.outer(active_index, active_index).to(device = self.device)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        print('Initialization loss:', torch.norm(self.A.data - A_rec, p = 'fro'))

    def forward(self):
        # The current matrix
        A = self.A

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

        # Diagonal left
        active_index = self.final_active_index.to(device = self.device)
        left_index = torch.outer(active_index, active_index)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index

        # Mother coefficients
        outer = torch.outer(1 - active_index, 1 - active_index)
        mother_coefficients = torch.reshape(D[outer == 1], (self.N - self.dim, self.N - self.dim))

        # Father coefficients
        outer = torch.outer(active_index, active_index)
        father_coefficients = torch.reshape(D[outer == 1], (self.dim, self.dim))

        # Mother wavelets
        mother_wavelets = torch.reshape(right[active_index == 0], (self.N - self.dim, self.N))

        # Father wavelets
        father_wavelets = torch.reshape(right[active_index == 1], (self.dim, self.N))

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        # Result
        return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets


# +------------------------+
# | Learnable MMF training |
# +------------------------+

def learnable_mmf_train(A, L, K, drop, dim, wavelet_indices, rest_indices, epochs = 10000, learning_rate = 1e-4, early_stop = True):
    model = Learnable_MMF(A, L, K, drop, dim, wavelet_indices, rest_indices)
    optimizer = Adagrad(model.parameters(), lr = learning_rate)

    # Training
    best = 1e9
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()

        A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

        loss = torch.norm(A - A_rec, p = 'fro')
        loss.backward()

        if epoch % 1000 == 0:
            print('---- Epoch', epoch, '----')
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

    loss = torch.norm(A - A_rec, p = 'fro')
    print('---- Final loss ----')
    print('Loss =', loss.item())

    return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets

