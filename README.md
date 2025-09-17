# BIREFRINGENCE FORECASTS

## Features

`BirefForecast` class allows quick computation of the Fisher Information matrix of cosmic birefringence and miscalibration angles given the noise spectra of the CMB experiment.
It is initialized with various parameters that describe the experimental setup, such as the frequency of the splits (`nu_GHzs`), their noise properties (`Nls_filenames`).
Then, several methods allows computation of Fisher matrix numerically (`.fisher()`) or with the analytical approach (`.analytical_fisher()`).
These analyses requires knowing the amplitude of the foregrounds in the studied masks. This is done by measuring the spectra of pysm3 frequency maps for several masks and then fitting foreground models on it. The code for this is not in this repository yet, only the results are in `fg_params.yaml`.

### `compute_Nls.py`

`BirefForecast` takes in input N_ell for noises. `compute_Nls.py` computes and plots N_ell for different experiments, taking into input rms, beam size, ell_knee etc.

### `number_of_angles.ipynb`

This notebook compares the errors on birefringence and miscalibration angles for different assumptions on miscalibration angles, e.g. it compares the error when fitting one single miscalibration angle for the whole experiment or one miscalibration angle per frequency.

### `a_dust_wide_CLEAN.py`

This code plots the error on birefringence angle as a function of the amplitude of the foregrounds.

### `error_experiments.ipynb`

This code plots forecasts MK method for different experiments and survey masks.

## Dependencies

This code uses [`numpy`](https://numpy.org), [`matplotlib`](https://matplotlib.org) and [`pspy`](https://github.com/simonsobs/pspy). Which are all installable through pip.

The paramfiles and most of the spectra files are in `pspy` format (i.e. are wrote/read through pspy.so_dict and pspy.so_spectra packages).
