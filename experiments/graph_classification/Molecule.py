# Molecule class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np
import Atom

class Molecule:
	def __init__(self, index, nAtoms, atomic_type):
		self.index = index
		self.nAtoms = nAtoms
		self.atomic_type = atomic_type
		assert self.nAtoms == len(self.atomic_type)

		self.atoms = []
		for i in range(self.nAtoms):
			self.atoms.append(Atom.Atom(index, self.atomic_type[i]))

	def set_neighbors(self, atom_index, nNeighbors, neighbors, weights):
		assert atom_index >= 0
		assert atom_index < self.nAtoms
		self.atoms[atom_index].set_neighbors(nNeighbors, neighbors, weights)

	def set_class(self, class_):
		self.class_ = class_

	def from_atomic_type_to_feature(self, all_atomic_type, sparse = False):
		nAtomicTypes = len(all_atomic_type)
		if sparse == True:
			x = []
			y = []
			v = []
			for i in range(self.nAtoms):
				index = all_atomic_type.index(self.atomic_type[i])
				assert index >= 0
				assert index < nAtomicTypes
				x.append(i)
				y.append(index)
				v.append(1.0)
			index_tensor = torch.LongTensor([x, y])
			value_tensor = torch.FloatTensor(v)
			self.feature = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nAtoms, nAtomicTypes]))
		else:
			self.feature = np.zeros((self.nAtoms, nAtomicTypes))
			for i in range(self.nAtoms):
				index = all_atomic_type.index(self.atomic_type[i])
				assert index >= 0
				assert index < nAtomicTypes
				self.feature[i, index] = 1.0
