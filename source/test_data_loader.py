from data_loader import *


# Kronecker product matrix
A = kron_def()
print('Kronecker:', A.size())


# Cycle graph
A = cycle_def()
print('Cycle:', A.size())


# +-----------------+
# | Citation graphs |
# +-----------------+

# Cora
adj, L_norm, features, labels, paper_ids, labels_list = citation_def(data_folder = '../data/', dataset = 'cora')
print('Cora:', L_norm.size())

# Citeseer
adj, L_norm, features, labels, paper_ids, labels_list = citation_def(data_folder = '../data/', dataset = 'citeseer')
print('Citeseer:', L_norm.size())

# WebKB
adj, L_norm, features, labels, paper_ids, labels_list = citation_def(data_folder = '../data/', dataset = 'WebKB')
print('WebKB:', L_norm.size())


# MNIST
A = mnist_def(data_folder = '../data/')
print('MNIST RBF kernel:', A.size())


# Minnesota road network
A = road_def(data_folder = '../data/')
print('Minnesota road network:', A.size())


# Karate club network
A = karate_def(data_folder = '../data/')
print('Karate club network:', A.size())


# Cayley tree
A, cayley_node_x, cayley_node_y, edges = cayley_def(cayley_order = 3, cayley_depth = 4)
print('Cayley tree:', A.size())

print('Done')
