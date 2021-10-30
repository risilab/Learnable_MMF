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

# Nystrom method
def nystrom_model(A, dim):
    N = A.size(0)
    perm = torch.randperm(N).detach().cpu().numpy()
    perm = perm[:dim]
    index = np.zeros(N)
    for k in range(perm.shape[0]):
        index[perm[k]] = 1
    index = torch.Tensor(index)
    outer = torch.outer(index, index)

    C = torch.transpose(A[index == 1], 0, 1)
    W = torch.reshape(A[outer == 1], (dim, dim))
    W_inverse = torch.matmul(torch.inverse(torch.matmul(torch.transpose(W, 0, 1), W) + 1e-2 * torch.eye(dim)), torch.transpose(W, 0, 1))
    A_rec = torch.matmul(torch.matmul(C, W_inverse), torch.transpose(C, 0, 1))
    norm = torch.norm(A - A_rec, p = 'fro')
    print('Error = ', norm)

    return A_rec, C, W_inverse
