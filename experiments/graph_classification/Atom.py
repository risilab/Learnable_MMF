# Atom class
class Atom:
	def __init__(self, index, atomic_type):
		self.index = index
		self.atomic_type = atomic_type

	def set_neighbors(self, nNeighbors, neighbors, weights):
		self.nNeighbors = nNeighbors
		self.neighbors = neighbors
		self.weights = weights
		assert self.nNeighbors == len(self.neighbors)
		assert self.nNeighbors == len(self.weights)
