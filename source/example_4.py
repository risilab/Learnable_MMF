import torch
from data_loader import *
from heuristics import *
from sparse_mmf_model import *

# Data loading
karate_laplacian = karate_def('../data/')

# Sparse MMF (heuristics to select indices)
print('--- Learnable MMF (sparse implementation) with indices selected by heuristics')
wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L = 26, K = 8, drop = 1, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = sparse_mmf_train(karate_laplacian, L = 26, K = 8, drop = 1, dim = 8, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 10000, learning_rate = 1e-3, early_stop = True)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

# Sparse MMF (heuristics to select indices, multiple wavelet indices for each rotation)
print('--- Learnable MMF (sparse implementation) with indices selected by heuristics for multiple wavelets per rotation')
wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(karate_laplacian.to_sparse(), L = 13, K = 8, drop = 2, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = sparse_mmf_train(karate_laplacian, L = 13, K = 8, drop = 2, dim = 8, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 10000, learning_rate = 1e-3, early_stop = True)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

