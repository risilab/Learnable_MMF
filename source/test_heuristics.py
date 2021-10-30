# For the large citation graphs
import networkx as nx
import scipy.sparse as sp
from input_data import load_data
from preprocessing import *

# For other datasets
from data_loader import *

# Heuristics
from heuristics import *

# +-------------+
# | Cayley tree |
# +-------------+

A_dense, cayley_node_x, cayley_node_y, edges = cayley_def(3, 4)
A_sparse = A_dense.to_sparse()

# Test 1
N = A_dense.size(0)
K = 16
L = 129
drop = 1
dim = 32

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
print('Done')

# Test 2
N = A_dense.size(0)
K = 16
L = 30
drop = 4
dim = 41

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

# +------------------+
# | Citeseer dataset |
# +------------------+

# Load data
adj, features = load_data('citeseer')

# Processing graph
adj_norm = preprocess_graph(adj)

# Processing features
features = sparse_to_tuple(features.tocoo())

# Torch sparse tensors
A_sparse = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
).coalesce()
A_dense = A_sparse.to_dense()

# Test 1
N = A_dense.size(0)
K = 32
L = 3000
drop = 1
dim = 327

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
print('Done')

# Test 1
N = A_dense.size(0)
K = 16
L = 300
drop = 10
dim = 327

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

# +--------------+
# | Cora dataset |
# +--------------+

# Load data
adj, features = load_data('cora')

# Processing graph
adj_norm = preprocess_graph(adj)

# Processing features
features = sparse_to_tuple(features.tocoo())

# Torch sparse tensors
A_sparse = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
).coalesce()
A_dense = A_sparse.to_dense()

# Test 1
N = A_dense.size(0)
K = 32
L = 2000
drop = 1
dim = 708

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
print('Done')

# Test 1
N = A_dense.size(0)
K = 16
L = 200
drop = 10
dim = 708

wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
print('Done')

wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)
print('Done')

print('All done')
