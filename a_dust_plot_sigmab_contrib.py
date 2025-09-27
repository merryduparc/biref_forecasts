from functions.biref_fisher_class import FisherBiref
from functions.analytical_fisher import analytical_fisher
from matplotlib import pyplot as plt
from pspy import so_spectra, so_dict
import numpy as np
from math import pi
from copy import deepcopy
import seaborn as sns

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif",     # Match LaTeX style
})

spectra_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
list_args = [
    "splits",
    "nu_GHzs",
    "bls_filenames",
    "noise_filenames",
]  # Args of fisher class that are a list

which_experiment = "Planck_HFI_no353"  # Planck_HFI, Planck_HFI_no353 or SO_MF_HF

ylims = {
    "Planck_HFI": [1e-5, 1e1],
    # "Planck_all_353": [1e-5, 1e3],
    "Planck_HFI_no353": [1e-5, 1e1],
    "SO_MF_HF": [1.2e-6, 1e0],
}


def args_combined(args_1: dict, args_2: dict) -> dict:
    args = deepcopy(args_1)
    for arg in list_args:
        arg_list = list(deepcopy(args_1[arg]))
        arg_list.extend(args_2[arg])
        args[arg] = arg_list
    return args


args_planck = so_dict.so_dict()
args_planck.read_from_file(f"paramfiles/{which_experiment}_args.dict")
args_planck["alphas_mapping"] = [0 for _ in args_planck["splits"]]
args_planck["combination_method"] = "comb_w_auto_EB"


def cov_thru_param(args, param_name, param_list):
    args_v = deepcopy(args)
    cov_list = []
    fisher_list = []
    for i, value in enumerate(param_list):
        args_v[param_name] = value
        fisher_class = FisherBiref(**args_v)
        fisher_list.append(fisher_class.analytical_fisher(0.0 / 180 * pi))
        cov_list.append(np.linalg.inv(fisher_list[i]) * ((180 / pi) ** 2))
    return np.array(fisher_list) / ((180 / pi) ** 2), np.array(cov_list)


linestyles = ["-" for _ in range(20)]

amps_list = np.logspace(-5.8, 5.8, 60)

amps_lists = [[amp for _ in range(4)] for amp in amps_list]
cov_lists = {}

cov_lists = {}
F_num, cov_lists["num"] = cov_thru_param(args_planck, "amp_dust", amps_lists)



cov_lists["1/F_bb"] = 1 / F_num[:, 0, 0]
cov_lists["1/F_fgfg"] = 1 / (F_num[:, 0, 0] - 2 * F_num[:, 1, 0] + F_num[:, 1, 1])
cov_lists["1/bb + 1/fgfg"] = cov_lists["1/F_bb"] + cov_lists["1/F_fgfg"]

cov_lists["1st corr"] = (
    2 * (F_num[:, 1, 0] - F_num[:, 0, 0]) * cov_lists["1/F_bb"] * cov_lists["1/F_fgfg"]
)
cov_lists["2nd corr"] = (
    cov_lists["1/bb + 1/fgfg"]
    * ((F_num[:, 1, 0] - F_num[:, 0, 0]) ** 2)
    * cov_lists["1/F_bb"]
    * cov_lists["1/F_fgfg"]
)

plt.figure()
fig, ax = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.01},
    sharex=True,
    dpi=150,
)
# print(cov_lists)
palette = {}
palette["num"] = "black"
palette["1/F_bb"] = "tab:green"
palette["1/F_fgfg"] = "blue"
palette["1/F_bfg"] = "purple"
palette["1/bb + 1/fgfg"] = "red"
palette["1st corr"] = "darkturquoise"
palette["2nd corr"] = "orange"

labels = {
    "num": 'Numerical',
    "1/F_bb": r"$1 / F_{\mathrm{CMB}\ \mathrm{CMB}}$",
    "1/F_fgfg": r"$1 / F_{\mathrm{FG}\ \mathrm{FG}}$",
    "1/F_bfg": r"$1 / F_{\mathrm{CMB}\ \mathrm{FG}}$",
    "1/bb + 1/fgfg": r"$1 / F_{\mathrm{CMB}\ \mathrm{CMB}} + 1 / F_{\mathrm{FG}\ \mathrm{FG}}$",
    "1st corr": "First correction",
    "2nd corr": "Second correction",
}

dotted_exp = ["1st corr", "2nd corr", "1/F_bfg"]

for i, (exp, cov_list) in enumerate(cov_lists.items()):
    if len(np.shape(cov_list)) == 1:
        ax[0].plot(
            amps_list,
            (cov_list[:]),
            label=labels[exp],
            color=palette[exp],
            alpha=1,
            linestyle=linestyles[i],
            linewidth=2.4,
        )
        ax[1].plot(
            amps_list,
            cov_list[:] / cov_lists["num"][:, 0, 0],
            label=f"{exp}/num",
            color=palette[exp],
            alpha=1,
            linestyle="-",
            linewidth=2.4,
        )
    elif len(np.shape(cov_list)) == 3:
        ax[0].plot(
            amps_list,
            (cov_list[:, 0, 0]),
            label=labels[exp],
            color=palette[exp],
            alpha=0.9,
            linestyle=linestyles[i],
            linewidth=4,
        )
        ax[1].plot(
            amps_list,
            cov_list[:, 0, 0] / cov_lists["num"][:, 0, 0],
            label=f"{exp}",
            color=palette[exp],
            alpha=1,
            linestyle="-",
            linewidth=2.4,
        )


ax[1].plot(
    amps_list,
    (
        cov_lists["num"][:, 0, 0]
        - cov_lists["1/bb + 1/fgfg"][:]
        - cov_lists["2nd corr"][:]
        - cov_lists["1st corr"][:]
    )
    / cov_lists["num"][:, 0, 0],
    label=f"Higher order residual",
    color="grey",
    alpha=1,
    linestyle="-",
    linewidth=2.4,
)

ax[0].plot(
    amps_list,
    (
        cov_lists["num"][:, 0, 0]
        - cov_lists["1/bb + 1/fgfg"][:]
        - cov_lists["2nd corr"][:]
        - cov_lists["1st corr"][:]
    ),
    label=f"Higher order residual",
    color="grey",
    alpha=1,
    linestyle="-",
    linewidth=2.4,
)

ax[0].legend(loc="upper right")
ax[0].set_ylabel(r"$\sigma(\beta)^2 f_{\rm sky}$", fontsize=20)
ax[0].grid(which="both", alpha=0.2)
ax[0].set_ylim(*ylims[which_experiment])
ax[0].set_xlim(1e-4, 0.1e6)
ax[0].semilogy()
ax[0].semilogx()

ax[1].set_xlabel(r"$a_{\mathrm{dust}}$", fontsize=20)
ax[1].set_ylabel(rf"Ratio to Numerical", fontsize=16)
ax[1].set_ylim(-2e-2, 1.02)
# ax[1].legend(loc="upper left")
# ax[1].semilogy()
ax[1].semilogx()


plt.tight_layout()
plt.savefig(f"plots/a_dust_sigma_b_{which_experiment}")
plt.savefig(f"plots/a_dust_sigma_b_{which_experiment}.pdf")

# ax[0].set_yscale("linear")
# ax[0].set_ylim(0, 0.3e0)
# ax[0].set_xlim(2e-1, 1e1)

# plt.savefig(f"plots/a_dust_sigma_b_{which_experiment}_zoom")
