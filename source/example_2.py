import torch
from data_loader import *
from learnable_mmf_smooth_wavelets_model import *
from drawing_utils import *

# Data loading
A, cayley_node_x, cayley_node_y, edges = cayley_def(cayley_order = 2, cayley_depth = 4)

# Learnable MMF with smooth wavelets
print('--- Learnable MMF with smooth wavelets')
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_smooth_wavelets_train(A, L = 42, K = 4, drop = 1, dim = 4, epochs = 20000, learning_rate = 1e-3, Lambda = 1, alpha = 0.5, early_stop = True)

# Visualize wavelets
for l in range(42):
    wavelet = mother_wavelets[l].unsqueeze(dim = 0).detach().cpu().numpy()
    draw_cayley(A.detach().cpu().numpy(), wavelet, cayley_node_x, cayley_node_y, edges, 'wavelet_' + str(l))

