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

EXPERIMENTS = ['Planck', 'SO LAT', 'SO LAT goal', 'SO SAT']

BANDS = {
    'Planck': ['030', '044', '070', '100', '143', '217', '353'],
    'SO LAT': ['027', '039', '093', '145', '225', '280'],
    'SO LAT goal': ['027', '039', '093', '145', '225', '280'],
    'SO SAT': ['027', '039', '093', '145', '225', '280'],
}


# RMS noises for SO LAT are for fsky=0.4, but we go up to fsky=0.56 when oberving galactic plane
# Should I increase rms noise to take this into account ?? or is the coverage even ?
fac_LAT = 1.
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
        '027': 61 / 60 * np.sqrt(2) * fac_LAT,
        '039': 30 / 60 * np.sqrt(2) * fac_LAT,
        '093': 5.3 / 60 * np.sqrt(2) * fac_LAT,
        '145': 6.6 / 60 * np.sqrt(2) * fac_LAT,
        '225': 15 / 60 * np.sqrt(2) * fac_LAT,
        '280': 35 / 60 * np.sqrt(2) * fac_LAT,
    },
    'SO LAT goal': {         # SO LAT numbers are in uK.arcmin
        '027': 44 / 60 * np.sqrt(2) * fac_LAT,
        '039': 23 / 60 * np.sqrt(2) * fac_LAT,
        '093': 3.8 / 60 * np.sqrt(2) * fac_LAT,
        '145': 4.1 / 60 * np.sqrt(2) * fac_LAT,
        '225': 10 / 60 * np.sqrt(2) * fac_LAT,
        '280': 25 / 60 * np.sqrt(2) * fac_LAT,
    },
    'SO SAT': {         # SO LAT numbers are in uK.arcmin
        '027': 35 / 60 * np.sqrt(2),
        '039': 21 / 60 * np.sqrt(2),
        '093': 2.6 / 60 * np.sqrt(2),
        '145': 3.3 / 60 * np.sqrt(2),
        '225': 6.3 / 60 * np.sqrt(2),
        '280': 16 / 60 * np.sqrt(2),
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
    'SO LAT goal': {
        '027': 7.4,
        '039': 5.1,
        '093': 2.2,
        '145': 1.4,
        '225': 1.0,
        '280': 0.9,
    },
    'SO SAT': {
        '027': 91,
        '039': 63,
        '093': 30,
        '145': 17,
        '225': 11,
        '280': 9,
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
    'SO LAT goal': {
        '027': 700,
        '039': 700,
        '093': 700,
        '145': 700,
        '225': 700,
        '280': 700,
    },
    'SO SAT': {
        '027': 30,
        '039': 30,
        '093': 50,
        '145': 50,
        '225': 70,
        '280': 100,
    },
}
L_KNEE['Planck'] = {band: None for band in BANDS['Planck']}

ALPHA_1fN = {
    'SO LAT': {
        '027': -1.4,
        '039': -1.4,
        '093': -1.4,
        '145': -1.4,
        '225': -1.4,
        '280': -1.4,
    },
    'SO LAT goal': {
        '027': -1.4,
        '039': -1.4,
        '093': -1.4,
        '145': -1.4,
        '225': -1.4,
        '280': -1.4,
    },
    'SO SAT': {
        '027': -2.4,
        '039': -2.4,
        '093': -2.5,
        '145': -3.0,
        '225': -3.0,
        '280': -3.0,
    },
}
ALPHA_1fN['Planck'] = {band: None for band in BANDS['Planck']}


SPECTRA_TO_COMPARE = {
    'Planck': {
        band: f'data/spectra/NPIPE_BIREF/Dls_{band}x{band}_noise.dat' for band in ['100', '143', '217', '353']
    },
    'SO LAT': {
        band: f'data/noise_model/Nls_SO{band}.dat' for band in BANDS['SO LAT']
    },
    'SO LAT goal': {
        band: None for band in BANDS['SO LAT']
    },
    'SO SAT': {
        band: None for band in BANDS['SO LAT']
    },
}

SPECTRA_TO_COMPARE['Planck']['030'] = None
SPECTRA_TO_COMPARE['Planck']['044'] = None
SPECTRA_TO_COMPARE['Planck']['070'] = None

COLORS = {
    'Planck': {band: f'C{i}' for i, band in enumerate(BANDS["Planck"])},
    'SO LAT': {band: f'C{i}' for i, band in enumerate(BANDS["SO LAT"])},
    'SO LAT goal': {band: f'C{i}' for i, band in enumerate(BANDS["SO LAT goal"])},
    'SO SAT': {band: f'C{i}' for i, band in enumerate(BANDS["SO SAT"])},
}

PLOTS_LIMS = {
    'Planck': {
        'xlims': (0, 3000),
        'ylims': (1, 50000)
    },
    'SO LAT': {
        'xlims': (0, 5000),
        'ylims': (.03, 3000)
    },
    'SO LAT goal': {
        'xlims': (0, 5000),
        'ylims': (.03, 3000)
    },
    'SO SAT': {
        'xlims': (0, 1500),
        'ylims': (.0003, 1000),
    },
}

def compute_Nls(rms:float, fwhm:float, l_knee:int, alpha:float=-1.4) -> dict[np.ndarray]:
    beam = hp.gauss_beam(np.deg2rad(fwhm / 60), lmax=10000)
    
    mean = {}
    Nls = {}
    for spec in utils.spectra_pspy:
        ls = np.arange(2, 10000)
        fac = ls * (ls + 1) / (2 * np.pi)   # Nls are in D_ell
        if (spec in ['EE', 'BB']):
            mean[spec] = np.full_like(ls, (rms ** 2) * ((np.pi / 180) ** 2), dtype=float) * fac
            if l_knee is not None:
                # From SO forecasts paper 1808.07445 (eq. 1)
                N_red_ratio = 1.
                mean[spec] *= (1 + N_red_ratio * (ls / l_knee)**alpha) 
        else:
            # Only polarization noise and assumes no cross noise
            mean[spec] = np.zeros_like(ls, dtype=float)
        Nls[spec] = mean[spec] / (beam[3:] ** 2)

    return ls, Nls

# Compute and plot noise curves
for exp in EXPERIMENTS:
    plt.figure()
    fig, ax = plt.subplots(dpi=150, figsize=(8, 4.5))
    for band in BANDS[exp]:
        rms = RMS[exp][band]
        fwhm = FWHM[exp][band]
        l_knee = L_KNEE[exp][band]
        alpha_1fN = ALPHA_1fN[exp][band]
        ls, Nls = compute_Nls(rms, fwhm, l_knee, alpha_1fN)
        so_spectra.write_ps(f'{SAVE_PATH}/Nl_{exp}_{band}.dat', ls, Nls, type='Dl', spectra=utils.spectra_pspy)
        ax.plot(ls, Nls['EE'], label=f'{exp} {band}', color=COLORS[exp][band])
        if SPECTRA_TO_COMPARE[exp][band] is not None:
            try:
                ls_comp, Nls_comp = so_spectra.read_ps(SPECTRA_TO_COMPARE[exp][band], spectra=utils.spectra_pspy)
                ax.plot(ls_comp, Nls_comp['EE'], label=f'{exp} {band}_ext', color=COLORS[exp][band], linewidth=1)
            except:
                print(f'{SPECTRA_TO_COMPARE[exp][band]} not found.')
    ax.set_xlim(*PLOTS_LIMS[exp]['xlims'])
    ax.set_ylim(*PLOTS_LIMS[exp]['ylims'])
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'$\ell$', fontsize=16)
    ax.set_ylabel(r'$\ell(\ell + 1) N_\ell / (2\pi) [\mu K^2]$', fontsize=16)
    ax.set_title(f'{exp}', fontsize=16)
    plt.savefig(f'plots/noises/Nls_{exp}.png')
    
    plt.clf()
