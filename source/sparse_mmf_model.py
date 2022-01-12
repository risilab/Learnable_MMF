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

# +------------+
# | Sparse MMF |
# +------------+

class Sparse_MMF(nn.Module):
    def __init__(self, A_dense, L, K, drop, dim, wavelet_indices, rest_indices, device = 'cpu'):
        super(Sparse_MMF, self).__init__()

        # Matrix
        self.A_dense = A_dense

        # Size of the matrix
        self.N = self.A_dense.size(0)
        assert self.A_dense.dim() == 2
        assert self.A_dense.size(1) == self.N

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

        # Initialization for the rotations
        self.all_O = torch.nn.ParameterList()

        self.all_rotation_indices = []
        self.padd_1_values = torch.FloatTensor([1 for i in range(self.N - self.K)])

        # The current matrix
        A = torch.Tensor(self.A_dense.data)

        active_index = torch.ones(self.N)
        for l in range(self.L):
            # Set the indices for this rotation
            indices = self.wavelet_indices[l] + self.rest_indices[l]
            indices.sort()
            assert len(indices) == self.K
            index = torch.zeros(self.N)
            for k in range(self.K):
                index[indices[k]] = 1
            
            # Outer product map
            outer = torch.outer(index, index)

            # Eigen-decomposition
            A_part = torch.matmul(A[index == 1], torch.transpose(A[index == 1], 0, 1))
            values, vectors = torch.eig(torch.reshape(A_part, (self.K, self.K)), True)
            
            # Rotation matrix
            O = torch.nn.Parameter(vectors.transpose(0, 1).data, requires_grad = True)
            self.all_O.append(O)

            # Create indices for the full rotation matrix
            row = []
            column = []
            for i in range(self.N):
                if i not in indices:
                    row.append(i)
                    column.append(i)
            for i in range(self.K):
                for j in range(self.K):
                    row.append(indices[i])
                    column.append(indices[j])
                    assert outer[indices[i], indices[j]] == 1
            rotation_indices = []
            rotation_indices.append(row)
            rotation_indices.append(column)
            self.all_rotation_indices.append(rotation_indices)
            if l % 100 == 0:
                print('Initialized rotation', l)

            # Create the full rotation matrix
            rotation_indices = torch.LongTensor(self.all_rotation_indices[l])
            rotation_values = torch.cat([self.padd_1_values, self.all_O[l].flatten()], dim = 0)

            U = torch.sparse.FloatTensor(rotation_indices, rotation_values, (self.N, self.N)).coalesce()
            
            # Rotate
            if l == 0:
                right = U.to_dense()
            else:
                right = torch.sparse.mm(U, right)

            # New A
            B = torch.sparse.mm(U, A)
            A = torch.sparse.mm(U, B.transpose(0, 1)).transpose(0, 1)
            
            # Drop the wavelet
            active_index[self.wavelet_indices[l]] = 0

        # Block diagonal left
        left_index = torch.outer(active_index, active_index).to(device = self.device)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        print('Initialization loss:', torch.norm(A_dense - A_rec, p = 'fro'))

    def forward(self):
        A = self.A_dense
        active_index = torch.ones(self.N)

        for l in range(self.L):
            # Create rotation matrix
            rotation_indices = torch.LongTensor(self.all_rotation_indices[l])
            rotation_values = torch.cat([self.padd_1_values, self.all_O[l].flatten()], dim = 0)

            U = torch.sparse.FloatTensor(rotation_indices, rotation_values, (self.N, self.N)).coalesce()

            # Rotate
            if l == 0:
                right = U.to_dense()
            else:
                right = torch.sparse.mm(U, right)

            B = torch.sparse.mm(U, A)
            A = torch.sparse.mm(U, B.transpose(0, 1)).transpose(0, 1)
            
            # Drop the wavelet
            active_index[self.wavelet_indices[l]] = 0

        # Block diagonal left
        left_index = torch.outer(active_index, active_index).to(device = self.device)
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

        return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets


# +---------------------+
# | Sparse MMF training |
# +---------------------+

# opt
# 'original': The original SGD with Cayley transform, without any additional library
# 'additional-sgd': SGD + momentum with Cayley transform, adopted from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform
# 'additional-adam': Adam optimizer with Cayley transform, adopted from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform

def sparse_mmf_train(A_dense, L, K, drop, dim, wavelet_indices, rest_indices, epochs = 10000, learning_rate = 1e-4, early_stop = True, opt = 'original', momentum = 0.9):
    model = Sparse_MMF(A_dense, L, K, drop, dim, wavelet_indices, rest_indices)
    
    if opt == 'original':
        # Original SGD with Cayley transform
        optimizer = Adagrad(model.parameters(), lr = learning_rate)
    else:
        # Use additional library
        import stiefel_optimizer

        # Create the list of trainable parameters
        param_g = []
        for l in range(L):
            param_g.append(model.all_O[l])

        # Create the dictionary of optimizer's hyperparameters
        dict_g = {
                'params': param_g,
                'lr': learning_rate,
                'momentum': momentum,
                'stiefel': True
        }

        # Create the optimizer
        if opt == 'additional-sgd':
            optimizer = stiefel_optimizer.SGDG([dict_g])
        else:
            optimizer = stiefel_optimizer.AdamG([dict_g])

    # Training
    best = 1e9
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()

        A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

        loss = torch.norm(A_dense - A_rec, p = 'fro')
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
            if opt == 'original':
                # Without any additional library
                for l in range(L):
                    X = torch.Tensor(model.all_O[l].data)
                    G = torch.Tensor(model.all_O[l].grad.data)
                    Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
                    tau = learning_rate
                    Y = torch.matmul(torch.matmul(torch.inverse(torch.eye(K) + tau / 2 * Z), torch.eye(K) - tau / 2 * Z), X)
                    model.all_O[l].data = Y.data
            else:
                # Use the optimizer from the additional library
                optimizer.step()

    # Return the result
    A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

    loss = torch.norm(A_dense - A_rec, p = 'fro')
    print('---- Final loss ----')
    print('Loss =', loss.item())

    return A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets


