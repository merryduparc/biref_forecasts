'''
Computes and plots Nls for given experiments, ONLY POLARIZATION NOISES.
Numbers for Planck are from Planck2018 paper 1807.06205


Eventually plots given spectra to compare Nls
'''

import os
from pspy import so_spectra
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from biref_forecasts import utils

# TODO Put all params in a yaml file ?

SAVE_PATH = 'data/Nls'
os.makedirs(SAVE_PATH, exist_ok=True)

EXPERIMENTS = ['Planck', 'SO LAT']

BANDS = {
    'Planck': ['030', '044', '070', '100', '143', '217', '353'],
    'SO LAT': ['027', '039', '093', '145', '225', '280'],
}

# RMS polarization noise in uK.deg
RMS = {
    'Planck': {
        '030': 3.5,
        '044': 4.0,
        '070': 5.0,
        '100': 1.96,
        '143': 1.17,
        '217': 1.75,
        '353': 7.31,
    },
    'SO LAT': {         # SO LAT numbers are in uK.arcmin
        '027': 71 / 60,
        '039': 36 / 60,
        '093': 8.0 / 60,
        '145': 10 / 60,
        '225': 22 / 60,
        '280': 54 / 60,
    },
}

# FWHM beam in arcmin
FWHM = {
    'Planck': {
        '030': 32.29,
        '044': 27.94,
        '070': 13.08,
        '100': 9.66,
        '143': 7.22,
        '217': 4.90,
        '353': 4.92,
    },
    'SO LAT': {
        '027': 7.4,
        '039': 5.1,
        '093': 2.2,
        '145': 1.4,
        '225': 1.0,
        '280': 0.9,
    },
}

L_KNEE = {
    'SO LAT': {
        '027': 700,
        '039': 700,
        '093': 700,
        '145': 700,
        '225': 700,
        '280': 700,
    },
}
L_KNEE['Planck'] = {band: None for band in BANDS['Planck']}


SPECTRA_TO_COMPARE = {
    'Planck': {
        band: f'data/spectra/NPIPE_BIREF/Dls_{band}x{band}_noise.dat' for band in ['100', '143', '217', '353']
    },
    'SO LAT': {
        band: None for band in BANDS['SO LAT']
    },
}
SPECTRA_TO_COMPARE['Planck']['030'] = None
SPECTRA_TO_COMPARE['Planck']['044'] = None
SPECTRA_TO_COMPARE['Planck']['070'] = None

COLORS = {
    'Planck': {band: f'C{i}' for i, band in enumerate(BANDS["Planck"])},
    'SO LAT': {band: f'C{i}' for i, band in enumerate(BANDS["SO LAT"])},
}

def compute_Nls(rms:float, fwhm:float, l_knee:int) -> dict[np.ndarray]:
    beam = hp.gauss_beam(np.deg2rad(fwhm / 60), lmax=10000)
    
    mean = {}
    Nls = {}
    for spec in utils.spectra_pspy:
        ls = np.arange(2, 10000)
        fac = ls * (ls + 1) / (2 * np.pi)
        if (spec in ['EE', 'BB']):
            mean[spec] = np.full_like(ls, (rms ** 2) * ((np.pi / 180) ** 2), dtype=float) * fac
            if l_knee is not None:
                # From SO forecasts paper 1808.07445 (eq. 1)
                # TODO put alpha and N_red_ratio as args
                alpha = -1.4
                N_red_ratio = 1.
                mean[spec] *= (1 + N_red_ratio * (ls / l_knee)**alpha) 
        else:
            # Only polarization noise and assumes no cross noise
            mean[spec] = np.zeros_like(ls, dtype=float)
        Nls[spec] = mean[spec] / (beam[3:] ** 2)

    return ls, Nls

for exp in EXPERIMENTS:
    plt.figure()
    fig, ax = plt.subplots(dpi=150, figsize=(8, 4.5))
    for band in BANDS[exp]:
        rms = RMS[exp][band]
        fwhm = FWHM[exp][band]
        l_knee = L_KNEE[exp][band]
        ls, Nls = compute_Nls(rms, fwhm, l_knee)
        so_spectra.write_ps(f'{SAVE_PATH}/Nl_{exp}_{band}.dat', ls, Nls, type='Dl', spectra=utils.spectra_pspy)
        ax.plot(ls, Nls['EE'], label=f'{exp}{band}', color=COLORS[exp][band])
        if SPECTRA_TO_COMPARE[exp][band] is not None:
            try:
                ls_comp, Nls_comp = so_spectra.read_ps(SPECTRA_TO_COMPARE[exp][band], spectra=utils.spectra_pspy)
                ax.plot(ls_comp, Nls_comp['EE'], label=f'{exp}{band}_ext', color=COLORS[exp][band], linewidth=1)
            except:
                print(f'{SPECTRA_TO_COMPARE[exp][band]} not found.')
    ax.set_xlim(0, 5000)
    ax.set_ylim(.1, 5000000)
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$\ell$', fontsize=16)
    ax.set_ylabel(r'$N_\ell$', fontsize=16)
    ax.set_title(f'{exp}', fontsize=16)
    plt.savefig(f'plots/Nls_{exp}.png')
    
    plt.clf()


