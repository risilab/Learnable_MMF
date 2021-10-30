# Graph class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np
import Molecule
import Atom

class Graph:
	def __init__(self, molecules, all_atomic_type):
		self.molecules = molecules

		self.nMolecules = len(self.molecules)
		self.nVertices = 0
		self.nEdges = 0
		for mol in range(self.nMolecules):
			self.nVertices += self.molecules[mol].nAtoms
			for atom in range(self.molecules[mol].nAtoms):
				self.nEdges += self.molecules[mol].atoms[atom].nNeighbors

		# Vertex indexing
		self.start_index = np.zeros((self.nMolecules + 1), dtype = np.int32)
		count = 0
		for mol in range(self.nMolecules):
			self.start_index[mol] = count
			count += self.molecules[mol].nAtoms
		assert count == self.nVertices
		self.start_index[self.nMolecules] = self.nVertices
		self.start_index = torch.from_numpy(self.start_index)

		# Adjacency matrix (total of all smaller molecules)
		self.adj = np.zeros((self.nVertices, self.nVertices), dtype = np.float32)

		# Edge indexing
		self.edges_tensor = np.zeros((self.nEdges, 2), dtype = np.float32)
		count = 0
		for mol in range(self.nMolecules):
			for atom in range(self.molecules[mol].nAtoms):
				u = self.start_index[mol] + atom
				for i in range(self.molecules[mol].atoms[atom].nNeighbors):
					v = self.start_index[mol] + self.molecules[mol].atoms[atom].neighbors[i]
					self.edges_tensor[count, 0] = u
					self.edges_tensor[count, 1] = v
					count += 1

					# Adjacency matrix
					self.adj[u, v] = 1.0
					self.adj[v, u] = 1.0
		assert count == self.nEdges
		self.edges_tensor = torch.from_numpy(self.edges_tensor)
		self.adj = torch.from_numpy(self.adj)

		# Feature indexing
		nAtomicTypes = len(all_atomic_type)
		x = []
		y = []
		v = []
		for mol in range(self.nMolecules):
			for i in range(self.molecules[mol].nAtoms):
				index = all_atomic_type.index(self.molecules[mol].atomic_type[i])
				x.append(self.start_index[mol] + i)
				y.append(index)
				v.append(1.0)
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		self.feature = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nVertices, nAtomicTypes]))

		# Label indexing
		self.label = []
		for mol in range(self.nMolecules):
			self.label.append(self.molecules[mol].class_)
		self.label = np.array(self.label)