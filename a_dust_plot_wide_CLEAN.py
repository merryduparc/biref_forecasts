from biref_forecasts.biref_fisher_class import FisherBiref
from matplotlib import pyplot as plt
from pspy import so_spectra, so_dict
import numpy as np
from math import pi
from copy import deepcopy
import seaborn as sns
import yaml

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif",     # Match LaTeX style
})

WHICH = ['SO', 'Planck']    # SO and/or Planck

spectra_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
list_args = ['splits', 'nu_GHzs', 'bls_filenames', 'Nls_filenames'] # Args of fisher class that are a list
def args_combined(args_1: dict, args_2: dict) -> dict:
    args = deepcopy(args_1)
    for arg in list_args:
        arg_list = list(deepcopy(args_1[arg]))
        arg_list.extend(args_2[arg])
        args[arg] = arg_list
    return args

### PLANCK
freqs_planck = ['100', '143', '217', '353']

args_planck = so_dict.so_dict()
args_planck.read_from_file('paramfiles/Planck_HFI_args.dict')

### PLANCK FREQS
args_planck_freqs = {}
for i, band in enumerate(freqs_planck):
    args_planck_freqs[band] = so_dict.so_dict()
    args_planck_freqs[band].read_from_file(f'paramfiles/Planck_{band}_args.dict')

# args_Planck_2f = args_combined(args_planck_freqs['100'], args_planck_freqs['217'])
# args_Planck_2f_auto = deepcopy(args_Planck_2f)
# args_Planck_2f_auto['combination_method'] = 'only_EB_auto'
# args_Planck_2f['alphas_mapping'] = [0, 0]
# args_Planck_3f = args_combined(args_Planck_2f, args_planck_freqs['143'])
# args_Planck_3f['alphas_mapping'] = [0, 0, 0]

### SO
freqs_so = ["093", "145", "225", "280"]

args_so = so_dict.so_dict()
args_so.read_from_file('paramfiles/SO_MF_HF_args.dict')
args_so['fsky'] = 1. # Fix this to on so you don;t have to multiply by sqrt(fsky) then

### SO freqs
args_SO_freqs = {}
for i, band in enumerate(freqs_so):
    args_SO_freqs[band] = so_dict.so_dict()
    args_SO_freqs[band].read_from_file(f'paramfiles/SO_{band}_args.dict')

def cov_thru_param(args, param_name, param_list):
    args_v = deepcopy(args)
    cov_list = []
    fisher_list = []
    for i, value in enumerate(param_list):
        args_v[param_name] = value
        fisher_class = FisherBiref(**args_v)
        fisher_list.append(fisher_class.fisher(0.0))
        cov_list.append(np.linalg.inv(fisher_list[i]) * ((180 / pi) ** 2))
    return np.array(fisher_list), np.array(cov_list)

linestyles = [
    "--",
    "-.",
    ":",
    (0, (3, 4, 1, 4)),
    (0, (1, 2, 1, 2, 1, 2)),
    (0, (3, 10, 1, 10)),
    "--",
    "-.",
]

amps_list = np.logspace(-4, 4, 35)
amps_list_planck = [[amp for _ in args_planck['splits']] for amp in amps_list]
amps_list_so = [[amp for _ in args_so['splits']] for amp in amps_list]
amps_list_2_freq = [[amp for _ in range(2)] for amp in amps_list]
amps_list_3_freq = [[amp for _ in range(3)] for amp in amps_list]

cov_lists = {}

if 'Planck' in WHICH:
    _, cov_lists["Planck"] = cov_thru_param(args_planck, "amp_dust", amps_list_planck)
    _, cov_lists["Planck 143GHz"] = cov_thru_param(args_planck_freqs['143'] , "amp_dust", amps_list_planck)
    # for i, band in enumerate(freqs_planck):
    #     _, cov_lists[f"Planck {band}"] = cov_thru_param(args_planck_freqs[f'{band}'], "amp_dust", amps_list_2_freq)
if 'SO' in WHICH:
    _, cov_lists["SO LAT"] = cov_thru_param(args_so, "amp_dust", amps_list_so)
    # for i, band in enumerate(freqs_so):
    #     _, cov_lists[f"SO LAT {band}"] = cov_thru_param(args_SO_freqs[f'{band}'], "amp_dust", amps_list_2_freq)
    

plt.figure()
fig, ax = plt.subplots(dpi=150, figsize=(8, 4.5))
freq_labels = {
    "Planck": [100, 143, 217, 353],
    "SO LAT": [93, 145, 225, 280],
}

with open(f'fg_params.yaml', "r") as f:
    fg_params: dict = yaml.safe_load(f)

measurements = {}
if 'Planck' in WHICH:
    measurements[r"Planck $f_{\rm sky}$=0.92"] = [fg_params['amp_dust']['full_sky'][0.92], 0.11 * np.sqrt(0.92)],
    measurements[r"Planck $f_{\rm sky}$=0.62"] = [fg_params['amp_dust']['full_sky'][0.62], 0.23 * np.sqrt(0.62)],

if 'SO' in WHICH:
    measurements[r"CO mask $f_{\rm sky}$=0.56"] = [fg_params['amp_dust']['SO'][0.56], np.sqrt(cov_thru_param(args_so, "amp_dust", [[3.867 for _ in range(4)]])[1][0][0, 0])],
    measurements[r"CO+Gal mask $f_{\rm sky}$=0.39"] = [fg_params['amp_dust']['SO'][0.39], np.sqrt(cov_thru_param(args_so, "amp_dust", [[0.886 for _ in range(4)]])[1][0][0, 0])],

measurements_marker = {
    r"Planck $f_{\rm sky}$=0.92": '*',
    r"Planck $f_{\rm sky}$=0.62": 'P',
    r"CO mask $f_{\rm sky}$=0.56": '*',
    r"CO+Gal mask $f_{\rm sky}$=0.39": 'P',
}

palette = {}
palette_Planck = sns.color_palette("Blues", n_colors=12)
palette_SO = sns.color_palette("Oranges", n_colors=12)
palette['Planck'] = palette_Planck[-1]
palette['Planck 143GHz'] = palette_Planck[-6]
palette['SO LAT'] = palette_SO[-1]
palette['SO LAT LS'] = palette_SO[-1]
for i in range(4):
    palette[f'Planck {freqs_planck[i]}'] = palette_Planck[i + 2]
    palette[f'SO LAT {freqs_so[i]}'] = palette_SO[i + 2]
measurements_colors = {
    r"Planck $f_{\rm sky}$=0.92": palette_Planck[-1],
    r"Planck $f_{\rm sky}$=0.62": palette_Planck[-1],
    r"CO mask $f_{\rm sky}$=0.56": palette_SO[-1],
    r"CO+Gal mask $f_{\rm sky}$=0.39": palette_SO[-1],
}

plot_alphas = False
for i, (exp, cov_list) in enumerate(cov_lists.items()):
    ax.plot(amps_list, np.sqrt(cov_list[:, 0, 0]), label=f"{exp}", color=palette[exp],)
    if plot_alphas:
        for j in range(np.shape(cov_list)[1] - 1):
            ax.plot(
                amps_list,
                np.sqrt(cov_list[:, j + 1, j + 1]),
                # label=f"a_{freq_labels[exp][j]} {exp}",
                linestyle=linestyles[j],
                # color=f"C{i}",
                color=palette[exp],
                alpha=0.15,
            )

sigmas_stat = {
    'Planck': 0.033 * np.sqrt(0.92),
    'SO LAT': 0.008 * np.sqrt(0.56),
}
for exp, sigma in sigmas_stat.items():
    ax.axhline(sigma, color=palette[exp], linestyle='--', label=fr'{exp} $\sigma_{{\alpha+\beta}}$', alpha=0.5, zorder=-10)

for meas_name, meas in measurements.items():
    ax.plot(*meas[0], marker=measurements_marker[meas_name], color=measurements_colors[meas_name], label=meas_name, linestyle='none', mfc='white', markersize=7)


ax.legend(loc=(1.01, 0.1))
ax.set_xlabel(r"$a_{\rm dust}$", fontsize=18)
ax.set_ylabel(r"$\sigma_\beta\ \sqrt{f_{\rm sky}}$", fontsize=18)
ax.grid(which="both", alpha=0.15)
ax.set_ylim(4e-3, 1e1)
ax.set_xlim(5e-3, 2.5e3)
ax.semilogy()
ax.semilogx()

plt.tight_layout()
plt.savefig(f'plots/a_dust_{'_'.join(WHICH)}_CLEAN.pdf')
plt.savefig(f'plots/a_dust_{'_'.join(WHICH)}_CLEAN')