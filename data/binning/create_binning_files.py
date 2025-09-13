from pspy import pspy_utils

for bin_size in [20, 40, 50, 100]:
    pspy_utils.create_binning_file(bin_size, 1000, lmax=10500, file_name=f'data/binning/binsize_{bin_size}.dat')

pspy_utils.create_binning_file(1, 10500, lmax=10500, file_name=f'data/binning/single_l_binning.dat')
