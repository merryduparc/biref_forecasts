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

EXPERIMENTS = ['Planck']

BANDS = {
    'Planck': ['030', '044', '070', '100', '143', '217', '353']
}

# RMS polarization noise in uK.arcmin
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
    # 'SO': {
    #     
    # }
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
    }
}

L_KNEE = {}
L_KNEE['Planck'] = {band: None for band in BANDS['Planck']}


SPECTRA_TO_COMPARE = {
    'Planck': {
        band: f'data/spectra/NPIPE_BIREF/Dls_{band}x{band}_noise.dat' for band in ['100', '143', '217', '353']
    }
}
SPECTRA_TO_COMPARE['Planck']['030'] = None
SPECTRA_TO_COMPARE['Planck']['044'] = None
SPECTRA_TO_COMPARE['Planck']['070'] = None

COLORS = {
    'Planck': {band: f'C{i}' for i, band in enumerate(BANDS["Planck"])}
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
            ls_comp, Nls_comp = so_spectra.read_ps(SPECTRA_TO_COMPARE[exp][band], spectra=utils.spectra_pspy)
            ax.plot(ls_comp, Nls_comp['EE'], label=f'{exp}{band}_ext', color=COLORS[exp][band], linewidth=1)
    ax.set_xlim(0, 5000)
    ax.set_ylim(10, 5000000)
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f'plots/Nls_{exp}.png')
    
    plt.clf()


