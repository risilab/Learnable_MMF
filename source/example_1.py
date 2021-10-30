import torch
from data_loader import *
from baseline_mmf_model import Baseline_MMF
from nystrom_model import nystrom_model
from heuristics import *
from learnable_mmf_model import *

# Data loading
karate_laplacian = karate_def('../data/')
N = karate_laplacian.size(0)

# Baseline (original) MMF
print('--- Baseline (original) MMF')
original_mmf = Baseline_MMF(N = N, L = N - 8, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = original_mmf(karate_laplacian)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

# Nystrom method
print('--- Nystrom method')
A_rec, C, W_inverse = nystrom_model(karate_laplacian, dim = 8)

# Learnable MMF (random indices)
print('--- Learnable MMF with random indices')
wavelet_indices, rest_indices = heuristics_random(karate_laplacian.to_sparse(), L = 26, K = 8, drop = 1, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian, L = 26, K = 8, drop = 1, dim = 8, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 10000, learning_rate = 1e-4, early_stop = True)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

# Learnable MMF (heuristics to select indices)
print('--- Learnable MMF with indices selected by heuristics')
wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L = 26, K = 8, drop = 1, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian, L = 26, K = 8, drop = 1, dim = 8, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 10000, learning_rate = 1e-3, early_stop = True)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

# Learnable MMF (heuristics to select indices, multiple wavelet indices for each rotation)
print('--- Learnable MMF with indices selected by heuristics for multiple wavelets per rotation')
wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(karate_laplacian.to_sparse(), L = 13, K = 8, drop = 2, dim = 8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian, L = 13, K = 8, drop = 2, dim = 8, wavelet_indices = wavelet_indices, rest_indices = rest_indices, epochs = 10000, learning_rate = 1e-3, early_stop = True)
print('Error =', torch.norm(karate_laplacian - A_rec, p = 'fro').item())

