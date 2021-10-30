import torch
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
from io import BytesIO

# Plot curve
def plot_curve(arr, title, xlabel, ylabel, file_name):
    plt.clf()
    plt.plot(arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)

# Draw matrix
def draw_mat(mat, file_name, title, dpi = 300):
    plt.clf()
    plt.matshow(mat)
    plt.colorbar()
    plt.title(title)
    plt.savefig(file_name, dpi = dpi)

# Draw wavelet
def draw_wavelet(wavelet, file_name, title):
    wavelet = wavelet.flatten()
    plt.clf()
    plt.plot(wavelet)
    plt.title(title)
    plt.savefig(file_name)

# Draw graph with wavelet
def draw_graph(A, wavelet, file_name):
    N = A.shape[0]
    R = 10
    nodes_x = []
    nodes_y = []
    nodes_z = []
    for k in range(N):
        alpha = 2 * np.pi * k / N
        x = R * np.cos(alpha)
        y = R * np.sin(alpha)
        z = 0
        nodes_x.append(x)
        nodes_y.append(y)
        nodes_z.append(z)

    plt.clf()
    fig = plt.figure(figsize = (R, R))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(nodes_x, nodes_y, nodes_z)

    for k1 in range(N):
        x1 = nodes_x[k1]
        y1 = nodes_y[k1]
        z1 = nodes_z[k1]
        for k2 in range(N):
            if A[k1, k2] == 0:
                continue
            x2 = nodes_x[k2]
            y2 = nodes_y[k2]
            z2 = nodes_z[k2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'b')

    for k in range(N):
        w = np.sum(wavelet[:, k])
        x1 = nodes_x[k]
        y1 = nodes_y[k]
        z1 = nodes_z[k]
        x2 = x1
        y2 = y1
        z2 = z1 + w
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'r')

    # PNG
    plt.savefig(file_name + '.pdf')

    # Close figures
    plt.close(fig)

# Draw Cayley graph with wavelets
def draw_cayley(A, wavelet, nodes_x, nodes_y, edges, file_name):
    N = A.shape[0]
    R = 10
    nodes_z = []
    for k in range(N):
        nodes_z.append(0)

    plt.clf()
    fig = plt.figure(figsize = (R, R))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(nodes_x, nodes_y, nodes_z)

    for edge in edges:
        u = edge[0]
        v = edge[1]
        x1 = nodes_x[u]
        y1 = nodes_y[u]
        z1 = nodes_z[u]
        x2 = nodes_x[v]
        y2 = nodes_y[v]
        z2 = nodes_z[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'b')

    for k in range(N):
        w = np.sum(wavelet[:, k])
        x1 = nodes_x[k]
        y1 = nodes_y[k]
        z1 = nodes_z[k]
        x2 = x1
        y2 = y1
        z2 = z1 + w
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'r')

    # PNG
    plt.savefig(file_name + '.pdf')

    # Close figures
    plt.close(fig)

