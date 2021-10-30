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

from mnist_dataset import mnist_dataset

# Kronecker graph
def kron_def():
    K1 = torch.Tensor(np.array(
        [
            [0, 1],
            [1, 1],
        ],
    ))
    K2 = torch.kron(K1, K1)
    K3 = torch.kron(K2, K1)
    K4 = torch.kron(K3, K1)
    K5 = torch.kron(K4, K1)
    K6 = torch.kron(K5, K1)
    K7 = torch.kron(K6, K1)
    K8 = torch.kron(K7, K1)
    K9 = torch.kron(K8, K1)
    A = K9
    print('Load the Kronecker graph')
    return A

# Cycle graph
def cycle_def():
    N = 64
    alpha = 0.1
    adj = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        adj[i, j] = 1
        adj[j, i] = 1
    adj = torch.Tensor(adj)
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Graph Laplacian
    L = torch.diag(D) - adj

    # Normalized graph Laplacian
    L_norm = torch.matmul(torch.matmul(DD, L), DD)

    # Option 1: Exponent
    # A = torch.exp(- alpha * L_norm)

    # Option 2: Just normalized graph Laplacian 
    A = L_norm

    # Option 3: Laplacian only
    # A = L

    print('Load the cycle graph')
    return A

# Original citation graph
def citation_def(data_folder, dataset):
    cites_fn = data_folder + '/' + dataset + '/' + dataset + '.cites'
    content_fn = data_folder + '/' + dataset + '/' + dataset + '.content'
    
    # Read content first
    paper_ids = []
    labels = []
    labels_list = []
    features = []
    with open(content_fn) as file:
        for line in file:
            words = line.split()
            paper_ids.append(words[0])
            label = words[len(words) - 1]
            if label not in labels_list:
                labels_list.append(label)
            labels.append(labels_list.index(label))
            feature = [float(w) for w in words[1:len(words) - 1]]
            features.append(feature)

    # Read the cites file
    N = len(paper_ids)
    adj = np.zeros((N, N))
    with open(cites_fn) as file:
        for line in file:
            words = line.split()
            assert len(words) == 2
            if words[0] in paper_ids and words[1] in paper_ids:
                id1 = paper_ids.index(words[0])
                id2 = paper_ids.index(words[1])
                adj[id1, id2] = 1
                adj[id2, id1] = 1

    adj = torch.Tensor(adj)
    
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Graph Laplacian
    L = torch.diag(D) - adj

    # Normalized graph Laplacian
    L_norm = torch.matmul(torch.matmul(DD, L), DD)

    features = torch.Tensor(np.array(features))
    labels = torch.LongTensor(np.array(labels))
    
    assert adj.size(0) == adj.size(1)
    assert adj.size(0) == len(paper_ids)
    assert features.size(0) == adj.size(0)
    assert labels.size(0) == adj.size(0)

    return adj, L_norm, features, labels, paper_ids, labels_list

# MNIST
def mnist_def(data_folder):
    data = mnist_dataset(directory = data_folder + '/mnist/', split = 'test', limit_data = -1)
    X = []
    y = []
    for c in range(10):
        count = 0
        i = 0
        while i < len(data.labels) and count < 100:
            if data.labels[i] == c:
                X.append(data.images[i])
                y.append(data.labels[i])
                count += 1
            i += 1
    print('Done sampling data')
    N = len(X)
    K = np.zeros((N, N))
    sigma = 10
    for i in range(N):
        x1 = X[i]
        for j in range(i + 1):
            x2 = X[j]
            d = np.linalg.norm(x1 - x2)
            K[i, j] = np.exp(- d * d / (2 * sigma * sigma))
            K[j, i] = K[i, j]
    A = torch.Tensor(K)
    print('Done kernel computation')
    return A

# Road of Minnesota
def road_def(data_folder):
    file = open(data_folder + '/minnesota/road-minnesota.mtx', 'r')
    for i in range(14):
        line = file.readline()
    words = file.readline().strip().split(' ')
    num_vertices = int(words[0])
    assert num_vertices == int(words[1])
    num_edges = int(words[2])
    print('Number of vertices:', num_vertices)
    print('Number of edges:', num_edges)
    adj = np.zeros((num_vertices, num_vertices))
    for edge in range(num_edges):
        words = file.readline().strip().split(' ')
        v1 = int(words[0]) - 1
        v2 = int(words[1]) - 1
        adj[v1, v2] = 1
        adj[v2, v1] = 1
    file.close()
    print('Done reading road in Minnesota dataset')

    adj = torch.Tensor(adj)
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Graph Laplacian
    L = torch.diag(D) - adj

    # Normalized graph Laplacian
    L = torch.matmul(torch.matmul(DD, L), DD)

    A = L
    print('Done computing the graph Laplacian')
    return A

# Karate network
def karate_def(data_folder):
    file = open(data_folder + '/karate/soc-karate.mtx', 'r')
    for i in range(23):
        line = file.readline()
    words = file.readline().strip().split(' ')
    num_vertices = int(words[0])
    assert num_vertices == int(words[1])
    num_edges = int(words[2])
    print('Number of vertices:', num_vertices)
    print('Number of edges:', num_edges)
    adj = np.zeros((num_vertices, num_vertices))
    for edge in range(num_edges):
        words = file.readline().strip().split(' ')
        v1 = int(words[0]) - 1
        v2 = int(words[1]) - 1
        adj[v1, v2] = 1
        adj[v2, v1] = 1
    file.close()
    print('Done reading the Karate club network')

    adj = torch.Tensor(adj)
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Graph Laplacian
    L = torch.diag(D) - adj

    # Normalized graph Laplacian
    L = torch.matmul(torch.matmul(DD, L), DD)

    A = L
    print('Done computing the graph Laplacian')
    return A

# Cayley graph
def cayley_def(cayley_order, cayley_depth):
    cayley_node_x = []
    cayley_node_y = []
    cayley_angle = []
    cayley_radius = []

    delta_R = 2

    edges = []

    # Center node
    cayley_node_x.append(0)
    cayley_node_y.append(0)
    cayley_angle.append(0)
    cayley_radius.append(0)
    N = 1
    R = 0

    # For all the depths
    for depth in range(cayley_depth):
        if depth == 0:
            degree = cayley_order + 1
        else:
            degree = cayley_order
        delta_N = N * degree

        node_x = []
        node_y = []
        angle = []
        radius = []
        selected = []

        for i in range(delta_N):
            a = 2 * np.pi * i / delta_N
            r = R + delta_R
            x = r * np.cos(a)
            y = r * np.sin(a)

            angle.append(a)
            radius.append(r)
            node_x.append(x)
            node_y.append(y)
            selected.append(False)
        
        for n in range(N):
            index1 = len(cayley_node_x) - N + n
            a1 = cayley_angle[index1]
            for d in range(degree):
                best = 1e9
                for i in range(delta_N):
                    if selected[i] == False:
                        a2 = angle[i]
                        if abs(a1 - a2) < best:
                            best = abs(a1 - a2)
                            index2 = i
                selected[index2] = True
                edges.append([index1, index2 + len(cayley_node_x)])

        cayley_node_x += node_x
        cayley_node_y += node_y
        cayley_angle += angle
        cayley_radius += radius

        N = delta_N
        R += delta_R

    # Create the adjacency matrix
    num_vertices = len(cayley_node_x)
    adj = np.zeros((num_vertices, num_vertices))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adj[u, v] = 1
        adj[v, u] = 1

    adj = torch.Tensor(adj)
    D = torch.sum(adj, dim = 0)
    DD = torch.diag(1.0 / torch.sqrt(D))

    # Graph Laplacian
    L = torch.diag(D) - adj

    # Normalized graph Laplacian
    L_norm = torch.matmul(torch.matmul(DD, L), DD)

    # Option 1: Exponent
    # A = torch.exp(- alpha * L_norm)

    # Option 2: Just normalized graph Laplacian 
    A = L_norm

    # Option 3: Laplacian only
    # A = L

    print('Done creating the Cayley graph')
    print('Cayley order =', cayley_order)
    print('Cayley depth =', cayley_depth)
    print('Number of vertices =', num_vertices)
    return A, cayley_node_x, cayley_node_y, edges

