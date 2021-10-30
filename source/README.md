## Documentation
Please check ```doc/document.pdf``` for a mathematical introduction and detailed tutorial for the usage.

## Built-in data loader
Built-in data loaders implemented in ```data_loader.py``` for:

* Synthetic datasets such as Kronecker product matrix, cycle graph and Cayley tree;

* Real datasets including citation graphs such as Cora, Citeseer and WebKB; RBF kernel gram matrix on MNIST digit images; Minnesota road network; and Karate club network.

Examples of data loading are in ```test_data_loader.py```.

## Baseline (original) MMF
The original (baseline) MMF [Kondor et al., 2014] is implemented in ```baseline_mmf_model.py``` with PyTorch module ```Baseline_MMF```. Script and training program (for large cases and other datasets) can be found at ```baseline_mmf_run.sh``` and ```baseline_mmf_run.py```.

## Learnable MMF
The learnable MMF applies Stiefel manifold optimization to find the optimal rotation matrices with arbitrary K. There are 3 algorithms (heuristics) implemented in ```heuristics.py``` to find K indices for each rotation: (1) ```heuristics_random```, (2) ```heuristics_k_neighbors_single_wavelet```, and (3) ```heuristics_k_neighbors_multiple_wavelets```. Indeed, users can input customized indices instead of the built-in ones.

The learnable MMF model is implemented in ```learnable_mmf_model.py```:

* ```Learnable_MMF```: The PyTorch module (network).

* ```learnable_mmf_train```: The training procedure to train the learnable MMF network, given the input hyperparamters including the indices selected by heuristics, number of epochs, and learning rate.

Short example code for usage is included in ```example_1.py```. Script and training program (for large cases and other datasets) can be found at ```learnable_mmf_run.sh``` and ```learnable_mmf_run.py```.

## Learnable MMF with smooth wavelets
The PyTorch network module and its training procedure for learnable MMF with smoothing loss function are in ```learnable_mmf_smooth_wavelets_model.py```. The API is exactly the same with other versions of MMFs. Script and training program (for large cases and other datasets) can be found at ```learnable_mmf_smooth_wavelets_run.*``` where ```* = sh, py```.

```example_2.py``` trains the learnable MMF with the smoothing loss and visualizes the smoothed wavelets on Cayley tree. ```example_3.py``` for the case of cycle graph. Drawing functions (outputing to ```.pdf``` files) are implemented in ```drawing_utils.py```.

## Sparse implementation of MMF
One problem with the dense implementation of MMFs is: storing sparse rotation matrices in dense format (i.e. NxN) is wasteful and computationally expensive (e.g., for matrix multiplication). Indeed, each rotation matrix has exactly a block of size KxK and the diagonal that are non-zero. We implement the sparse version by storing rotation matrices in PyTorch's COO sparse format and replace dense matrix multiplication operations by the sparse ones. The model (```Sparse_MMF```) and training procedure (```sparse_mmf_train```) are implemented in ```sparse_mmf_model.py```. Here is an example of usage for the Karate club network in Python interactive command line (check ```example_4.py``` also). Script and training program (for large cases and other datasets) can be found at ```sparse_mmf_run.sh``` and ```sparse_mmf_run.py```.
