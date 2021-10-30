import torch
from data_loader import *
from learnable_mmf_smooth_wavelets_model import *
from drawing_utils import *

# Data loading
A = cycle_def()

# Learnable MMF with smooth wavelets
print('--- Learnable MMF with smooth wavelets')
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_smooth_wavelets_train(A, L = 60, K = 4, drop = 1, dim = 4, epochs = 20000, learning_rate = 1e-3, Lambda = 1, alpha = 0.5, early_stop = True)

# Visualize wavelets
for l in range(60):
    wavelet = mother_wavelets[l].unsqueeze(dim = 0).detach().cpu().numpy()
    draw_graph(A.detach().cpu().numpy(), wavelet, 'wavelet_' + str(l))

