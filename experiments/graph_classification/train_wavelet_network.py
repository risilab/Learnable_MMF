import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import time

import numpy as np
import scipy.io
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
from io import BytesIO

from Dataset import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Wavelet Network')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = '../../data/', help = 'Data folder')
    parser.add_argument('--dataset', '-dataset', type = str, default = '.', help = 'Graph kernel benchmark')
    parser.add_argument('--adjs', '-adjs', type = str, default = '.', help = 'Adjacencies')
    parser.add_argument('--laplacians', '-laplacians', type = str, default = '.', help = 'Laplacians')
    parser.add_argument('--mother_wavelets', '-mother_wavelets', type = str, default = '.', help = 'Mother wavelets')
    parser.add_argument('--father_wavelets', '-father_wavelets', type = str, default = '.', help = 'Father wavelets')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--split', '-split', type = int, default = 0, help = 'Split index from 0 to 4')
    parser.add_argument('--num_layers', '-num_layers', type = int, default = 4, help = 'Number of layers')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 100, help = 'Hidden dimension')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
log_name = args.dir + "/" + args.name + ".log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

print(args.name)
print(args.dir)

# Data loader
class torch_dataset(Dataset):
    def __init__(self, adjs, laplacians, mother_wavelets, father_wavelets, features, labels, num_classes, indices):
        self.max_size = 0
        for sample in adjs:
            if sample is None:
                continue
            if sample.size(0) > self.max_size:
                self.max_size = sample.size(0)
        
        self.adjs = []
        self.laplacians = []
        self.mother_wavelets = []
        self.father_wavelets = []
        self.features = []
        self.labels = []
        self.num_classes = num_classes

        for index in indices:
            if adjs[index] is None:
                continue
            self.adjs.append(adjs[index])
            self.laplacians.append(laplacians[index])
            self.mother_wavelets.append(mother_wavelets[index])
            self.father_wavelets.append(father_wavelets[index])
            self.features.append(features[index])
            self.labels.append(labels[index])

        self.num_samples = len(self.adjs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # One hot label
        one_hot = torch.zeros(self.num_classes)
        one_hot[self.labels[idx]] = 1

        # Bases
        if self.mother_wavelets[idx] is not None:
            mother = self.mother_wavelets[idx].squeeze(dim = 0)
            father = self.father_wavelets[idx].squeeze(dim = 0)
            bases = torch.cat([mother, father], dim = 0)
        else:
            bases = self.laplacians[idx]

        # Pad with zeros
        matrices = self.pad_1(torch.DoubleTensor(self.adjs[idx].double()))
        bases = self.pad_1(bases)
        features = self.pad_2(torch.DoubleTensor(self.features[idx].astype(np.double)))

        # Sample
        sample = {
            'matrices': matrices,
            'bases': bases,
            'features': features,
            'labels': one_hot
        }
        return sample

    def pad_1(self, inp):
        n = inp.size(0)
        out = torch.zeros(self.max_size, self.max_size)
        out[:n, :n] = inp
        return out

    def pad_2(self, inp):
        n = inp.size(0)
        f = inp.size(1)
        out = torch.zeros(self.max_size, f)
        out[:n, :] = inp
        return out

# Wavelet Network
class Wavelet_Network(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, device = 'cpu', **kwargs):
        super(Wavelet_Network, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.input_layer_1 = nn.Linear(self.input_dim, self.hidden_dim).to(device = self.device)
        self.input_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device = self.device)

        self.hidden_layers = nn.ModuleList()
        for layer in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim).to(device = self.device))

        self.output_layer_1 = nn.Linear(self.hidden_dim * (self.num_layers + 1), self.hidden_dim).to(device = self.device)
        self.output_layer_2 = nn.Linear(self.hidden_dim, self.output_dim).to(device = self.device)

    def forward(self, W, f):
        all_hiddens = []

        # Input
        h = torch.tanh(self.input_layer_1(f))
        h = torch.tanh(self.input_layer_2(h))
        all_hiddens.append(h)

        # Hiddens
        for layer in range(self.num_layers):
            h = torch.tanh(torch.matmul(W.transpose(1, 2), self.hidden_layers[layer](torch.matmul(W, h))))
            all_hiddens.append(h)

        # Output
        concat = torch.cat(all_hiddens, dim = 2)
        output_1 = torch.tanh(self.output_layer_1(concat))
        output_2 = torch.sum(output_1, dim = 1)
        return torch.nn.functional.softmax(self.output_layer_2(output_2))

# Data loader
data_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.dat'
meta_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.meta'

data = Dataset(data_fn, meta_fn)
num_data = data.nMolecules

features = []
for sample in range(num_data):
    features.append(data.molecules[sample].feature)
assert len(features) == num_data
print(features[0])
num_features = features[0].shape[1]
print('Number of atomic features:', num_features)

print('Number of classes:', data.nClasses)

# Load the adjs and laplacians
adjs = torch.load(args.adjs)
laplacians = torch.load(args.laplacians)

# Load wavelet coefficients
mother_wavelets = torch.load(args.mother_wavelets)
father_wavelets = torch.load(args.father_wavelets)

assert len(mother_wavelets) == num_data
assert len(father_wavelets) == num_data

percents = []
for wavelet in mother_wavelets:
    if wavelet is None:
        continue
    percent = torch.count_nonzero(wavelet) / torch.numel(wavelet)
    percents.append(percent)
print('Average sparsity:', np.mean(percents) * 100)

# Train/Test split
def load_indices(file_name):
    file = open(file_name, 'r')
    file.readline()
    num_samples = int(file.readline())
    file.readline()
    indices = []
    for i in range(num_samples):
        indices.append(int(file.readline()))
    file.close()
    return indices

train_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.train.' + str(args.split)
test_fn = args.data_folder + '/' + args.dataset + '/' + args.dataset + '.test.' + str(args.split)

train_indices = load_indices(train_fn)
test_indices = load_indices(test_fn)

assert len(train_indices) + len(test_indices) == num_data

# Datasets
train_dataset = torch_dataset(adjs, laplacians, mother_wavelets, father_wavelets, features, data.all_labels, data.nClasses, train_indices)
test_dataset = torch_dataset(adjs, laplacians, mother_wavelets, father_wavelets, features, data.all_labels, data.nClasses, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

# Init model and optimizer
model = Wavelet_Network(
        num_layers = args.num_layers, 
        input_dim = num_features, 
        hidden_dim = args.hidden_dim,
        output_dim = data.nClasses
)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

# Compute accuracy
def accuracy(predict, target):
    predict = torch.argmax(predict, dim = 1)
    target = torch.argmax(target, dim = 1)
    num_samples = predict.size(0)
    acc = 0
    for i in range(num_samples):
        if predict[i] == target[i]:
            acc += 1
    acc /= num_samples
    return acc

# Train model
best_acc = 0
for epoch in range(args.num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    for batch_idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        matrices = data['matrices'].to(device = device)
        bases = data['bases'].to(device = device)
        features = data['features'].float().to(device = device)
        target = data['labels'].to(device = device)

        predict = model(bases, features)

        loss = F.binary_cross_entropy(predict.view(-1), target.view(-1), reduction = 'mean')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 100 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    avg_loss = total_loss / nBatch
    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Testing
    t = time.time()
    model.eval()

    all_predict = []
    all_target = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            matrices = data['matrices'].to(device = device)
            bases = data['bases'].to(device = device)
            features = data['features'].float().to(device = device)
            target = data['labels'].to(device = device)

            predict = model(bases, features)

            all_predict.append(predict)
            all_target.append(target)
        
        # Compute accuracies
        all_predict = torch.cat(all_predict, dim = 0)
        all_target = torch.cat(all_target, dim = 0)

        acc = accuracy(all_predict, all_target)
        print('Test accuracy:', acc)

        print("Test time =", "{:.5f}".format(time.time() - t))
        LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

    if acc > best_acc:
        best_acc = acc
        print('Current best accuracy updated:', best_acc)
        LOG.write('Current best accuracy updated: ' + str(best_acc) + '\n')

        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")

print('Best accuracy:', best_acc)
LOG.write('Best accuracy: ' + str(best_acc) + '\n')
LOG.close()
