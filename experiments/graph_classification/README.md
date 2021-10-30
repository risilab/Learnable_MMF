## Documentation
Please check ```doc/document.pdf``` for details.

## Training the MMF wavelet networks
We also tested our wavelet networks on standard graph classification benchmarks including four bioinformatics datasets: (1) MUTAG, (2) PTC, (3) PROTEINS or DD, (4) NCI1. First, we need to run MMF to factorize all normalized graph Laplacians in the dataset to get the wavelet basis (for each individual molecule):

* Baseline MMF: Script to run and program are ```baseline_mmf_basis.sh``` and ```baseline_mmf_basis.py```.

* Learnable MMF: Script to run and program are ```learnable_mmf_basis.sh``` and ```learnable_mmf_basis.py```.

Second, based on wavelet basis that we found, we train our wavelet networks (with spectral convolution): script and program are ```train_wavelet_network.sh``` and ```train_wavelet_network.py```, respectively. The PyTorch network module is ```Wavelet_Network``` in the Python program.
