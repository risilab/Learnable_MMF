# Dataset class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np

import Graph
import Molecule
import Atom

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class Dataset:
	def __init__(self, data_fn, meta_fn):
		self.data_fn = data_fn
		self.meta_fn = meta_fn

		self.load_meta()
		self.load_data()
		self.from_atomic_type_to_feature()

	def load_meta(self):
		file = open(self.meta_fn, "r")
		file.readline()
		self.nMolecules = int(file.readline().strip())
		file.readline()
		self.total_nAtoms = int(file.readline().strip())
		file.readline()
		self.total_nBonds = int(file.readline().strip())
		file.readline()
		self.max_nAtoms = int(file.readline().strip())
		file.readline()
		self.min_nAtoms = int(file.readline().strip())
		file.readline()
		self.max_degree = int(file.readline().strip())
		file.readline()
		self.min_degree = int(file.readline().strip())
		file.readline()
		self.max_weight = float(file.readline().strip())
		file.readline()
		self.min_weight = float(file.readline().strip())
		file.readline()
		self.density = float(file.readline().strip())
		file.readline()
		self.nAtomicTypes = int(file.readline().strip())
		file.readline()
		self.all_atomic_type = file.readline().strip().split(' ')
		assert self.nAtomicTypes == len(self.all_atomic_type)
		file.readline()
		self.nClasses = int(file.readline().strip())
		file.readline()
		self.classes = file.readline().strip().split(' ')
		assert self.nClasses == len(self.classes)
		file.close()

	def load_data(self):
		file = open(self.data_fn, "r")
		file.readline()
		assert self.nMolecules == int(file.readline().strip())
		self.molecules = []
		self.all_labels = []
		for mol in range(self.nMolecules):
			file.readline()
			file.readline()
			nAtoms = int(file.readline().strip())
			file.readline()
			atomic_type = file.readline().strip().split(' ')
			assert nAtoms == len(atomic_type)
			for i in range(nAtoms):
				assert (atomic_type[i] in self.all_atomic_type)
			self.molecules.append(Molecule.Molecule(mol, nAtoms, atomic_type))
			for i in range(nAtoms):
				file.readline()
				nNeighbors = int(file.readline().strip())
				file.readline()
				if nNeighbors > 0:
					neighbors = [int(element) for element in file.readline().strip().split(' ')]
				else:
					neighbors = []
					file.readline()
				file.readline()
				if nNeighbors > 0:
					weights = [float(element) for element in file.readline().strip().split(' ')]
				else:
					weights = []
					file.readline()
				self.molecules[mol].set_neighbors(i, nNeighbors, neighbors, weights)
			file.readline()
			class_ = file.readline().strip()
			assert (class_ in self.classes)
			self.molecules[mol].set_class(self.classes.index(class_))
			self.all_labels.append(self.classes.index(class_))
		self.all_labels = np.array(self.all_labels)
		file.close()

	def from_atomic_type_to_feature(self):
		for mol in range(self.nMolecules):
			self.molecules[mol].from_atomic_type_to_feature(self.all_atomic_type)

	def load_indices(self, file_fn):
		file = open(file_fn, "r")
		indices = []
		file.readline()
		nExamples = int(file.readline().strip())
		file.readline()
		for i in range(nExamples):
			index = int(file.readline().strip())
			assert index >= 0
			assert index < self.nMolecules
			indices.append(index)
		file.close()
		return np.array(indices)
