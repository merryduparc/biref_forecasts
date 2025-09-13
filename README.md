# BIREFRINGENCE FORECASTS

## Features

The `BirefForecast` class allows quick computation of the Fisher Information matrix of cosmic birefringence and miscalibration angles given the noise spectra of the CMB experiment.

## Dependencies

This code uses [`numpy`](https://numpy.org), [`matplotlib`](https://matplotlib.org) and [`pspy`](https://github.com/simonsobs/pspy). Which are all installable through pip.

The paramfiles and most of the spectra files are made with `pspy` (i.e. are wrote/read through pspy.so_dict and pspy.so_spectra packages).
The `.ipynb` notebook should allow users to get their own paramfiles and to adjust their spectra in the right file format.
