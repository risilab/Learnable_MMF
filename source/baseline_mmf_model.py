import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader

import numpy as np

# Baseline MMF
class Baseline_MMF(nn.Module):
    def __init__(self, N, L, dim, device = 'cpu'):
        super(Baseline_MMF, self).__init__()

        # Matrix size
        self.N = N

        # Number of resolutions
        self.L = L

        # Size of the Jacobian rotation matrix is always K = 2
        self.K = 2

        # Number of rows/columns to drop in each iteration is always drop = 1
        self.drop = 1

        # Number of left rows/columns at the end
        self.dim = dim
        assert self.dim == self.N - self.L * self.drop

        # Device
        self.device = device

    def forward(self, A):
        # Active index
        active_index = torch.ones(self.N)

        for l in range(self.L):
            active_list = np.array([i for i in range(self.N) if active_index[i] == 1])
            perm = torch.randperm(active_list.shape[0]).detach().cpu().numpy()
            assert active_list.shape[0] == perm.shape[0]

            # Select the first vertex/node
            first_node = active_list[perm[0]]

            # Select the second vertex/node
            best_epsilon = 1e9
            for v in active_list.tolist():
                if v < first_node:
                   i = v
                   j = first_node
                elif v > first_node:
                    i = first_node
                    j = v
                else:
                    continue
                A_sub = (A[[i, j], :]).detach().cpu().numpy()
                B = np.matmul(A_sub, np.transpose(A_sub))
                A_sub = A_sub[:, [i, j]]
                values, vectors = np.linalg.eig(B)
                O = vectors.transpose()
                term_1 = np.matmul(np.matmul(O, A_sub), O.transpose())
                term_2 = np.matmul(np.matmul(O, B), O.transpose())
                epsilon = 2 * term_1[1, 0] + 2 * term_2[1, 1]
                if epsilon < best_epsilon:
                    best_epsilon = epsilon
                    second_node = v

            # Selected 2 indices
            selected_index = torch.zeros(self.N)
            selected_index[first_node] = 1
            selected_index[second_node] = 1
            selected_index = selected_index.to(device = self.device)

            # Drop index for wavelet
            drop_index = torch.zeros(self.N)
            if first_node < second_node:
                ind = second_node
            else:
                ind = first_node
            active_index[ind] = 0
            drop_index[ind] = 1
            drop_index = drop_index.to(device = self.device)

            # Outer product map
            outer = torch.outer(selected_index, selected_index)

            # Eigen-decomposition
            A_sub = torch.matmul(A[selected_index == 1], torch.transpose(A[selected_index == 1], 0, 1))
            values, vectors = torch.eig(A_sub, True)

            # Rotation matrix
            O = vectors.transpose(0, 1).data

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
        left_index = active_index.to(device = self.device)
        left_index = torch.outer(left_index, left_index)
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

