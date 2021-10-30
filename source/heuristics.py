import torch
import numpy as np

# Find all neighbor nodes and similarity scores
def construct_neighbors(A_sparse):
    N = A_sparse.size(0)
    indices = A_sparse.indices()
    values = A_sparse.values()
    num_nonzeros = values.size(0)

    neighbors = []
    scores = []
    for i in range(N):
        left = 0
        right = num_nonzeros - 1
        pos1 = num_nonzeros
        while left <= right:
            mid = (left + right) // 2
            if indices[0][mid] == i:
                if pos1 > mid:
                    pos1 = mid
                right = mid - 1
                continue
            if indices[0][mid] < i:
                left = mid + 1
            else:
                right = mid - 1

        neighbor = []
        score = []
        if pos1 < num_nonzeros:
            j = pos1
            while j < num_nonzeros:
                if indices[0][j] != i:
                    break
                if indices[1][j] != i:
                    neighbor.append(indices[1][j].item())
                    score.append(values[j].item())
                j += 1

            for u in range(len(neighbor)):
                v = u + 1
                while v < len(neighbor):
                    if score[u] < score[v]:
                        temp = neighbor[u]
                        neighbor[u] = neighbor[v]
                        neighbor[v] = temp
                        temp = score[u]
                        score[u] = score[v]
                        score[v] = temp
                    v += 1

        neighbors.append(neighbor)
        scores.append(score)
    return neighbors, scores

# Verify the correctness of hierarchy
def verify_hierarchy(wavelet_indices, rest_indices, N, L, K, drop, dim):
    assert N == L * drop + dim
    assert drop < K
    assert dim >= K
    
    assert len(wavelet_indices) == L
    assert len(rest_indices) == L

    is_dropped = np.zeros(N)
    for i in range(L):
        assert len(wavelet_indices[i]) == drop
        assert len(wavelet_indices[i]) + len(rest_indices[i]) == K
        for j in range(drop):
            node = wavelet_indices[i][j]
            assert is_dropped[node] == 0
            is_dropped[node] = 1
        for j in range(K - drop):
            node = rest_indices[i][j]
            assert wavelet_indices[i].count(node) == 0
            assert rest_indices[i].count(node) == 1

# +----------------------------------------------+
# | Heuristics: K-neighbors for a single wavelet |
# +----------------------------------------------+

def heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim):
    N = A_sparse.size(0)
    assert drop == 1
    assert N == L * drop + dim
    assert drop < K
    assert dim >= K

    # Find all neighbor nodes and similarity scores
    neighbors, scores = construct_neighbors(A_sparse)

    # Find the wavelet indices & rest indices
    perm = np.random.permutation(N)

    wavelet_indices = []
    rest_indices = []

    for i in range(L):
        node = perm[i]
        wavelet_indices.append([node])
        indices = []
        for j in range(len(neighbors[node])):
            indices.append(neighbors[node][j])
            if len(indices) == K - 1:
                break
        while len(indices) < K - 1:
            while True:
                random = np.random.randint(N)
                if random in indices:
                    continue
                if random == node:
                    continue
                indices.append(random)
                break
        rest_indices.append(indices)

    verify_hierarchy(wavelet_indices, rest_indices, N, L, K, drop, dim)
    return wavelet_indices, rest_indices


# +-----------------------------------------------+
# | Heuristics: K-neighbors for multiple wavelets |
# +-----------------------------------------------+

def heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim):
    N = A_sparse.size(0)
    assert N == L * drop + dim
    assert dim >= K

    # Find all neighbor nodes and similarity scores
    neighbors, scores = construct_neighbors(A_sparse)

    # Find the wavelet indices & rest indices
    perm = np.random.permutation(N)

    # Mark if a node is removed
    is_dropped = np.zeros(N)

    wavelet_indices = []
    rest_indices = []

    i = 0
    while i < N:
        node = perm[i]
        if is_dropped[node] == 1:
            i = i + 1
            continue

        # Find the wavelet indices
        indices = [node]
        is_dropped[node] = 1

        if len(indices) < drop:
            for j in range(len(neighbors[node])):
                v = neighbors[node][j]
                if v not in indices:
                    if is_dropped[v] == 0:
                        indices.append(v)
                        is_dropped[v] = 1
                        if len(indices) == drop:
                            break
        
        if len(indices) < drop:
            j = i + 1
            while j < N:
                v = perm[j]
                if is_dropped[v] == 0:
                    indices.append(v)
                    is_dropped[v] = 1
                    if len(indices) == drop:
                        break
                j = j + 1

        wavelet_indices.append(indices)

        # Find the rest indices
        indices = []

        for j in range(len(neighbors[node])):
            v = neighbors[node][j]
            if v not in indices:
                if is_dropped[v] == 0:
                    indices.append(v)
                    if len(indices) == K - drop:
                        break

        if len(indices) < K - drop:
            j = i + 1
            while j < N:
                v = perm[j]
                if is_dropped[v] == 0 and v not in indices:
                    indices.append(v)
                    if len(indices) == K - drop:
                        break
                j = j + 1

        rest_indices.append(indices)

        i = i + 1
        if len(wavelet_indices) == L:
            break

    # Verify the quality
    verify_hierarchy(wavelet_indices, rest_indices, N, L, K, drop, dim)

    return wavelet_indices, rest_indices


# +-----------------------------------------------+
# | Heuristics: Completely random (no heuristics) |
# +-----------------------------------------------+

def heuristics_random(A_sparse, L, K, drop, dim):
    N = A_sparse.size(0)
    assert N == L * drop + dim
    assert dim >= K

    # Find all neighbor nodes and similarity scores
    neighbors, scores = construct_neighbors(A_sparse)

    # Find the wavelet indices & rest indices
    wavelet_perm = np.random.permutation(N)

    wavelet_indices = []
    rest_indices = []
    u = 0
    for i in range(L):
        # Wavelet indices first
        indices = []
        while u < N:
            node = wavelet_perm[u]
            indices.append(node)
            u = u + 1
            if len(indices) == drop:
                break
        wavelet_indices.append(indices)

        # Rest indices
        rest_perm = np.random.permutation(N)
        indices = []
        v = 0
        while v < N:
            node = rest_perm[v]
            if node not in wavelet_indices[i]:
                indices.append(node)
                if len(indices) == K - drop:
                    break
            v = v + 1
        rest_indices.append(indices)

    # Verify the quality
    verify_hierarchy(wavelet_indices, rest_indices, N, L, K, drop, dim)

    return wavelet_indices, rest_indices
