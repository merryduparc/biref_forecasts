import numpy as np
from scipy.optimize import curve_fit
from functions.noise_functions import bin_array
from math import pi

h = 6.62607015e-34  # Planck's constant in J s
k_B = 1.380649e-23  # Boltzmann's constant in J/K
c = 2.99792458e8

T_D = 20  # d0 : 20K       ACT : 19.6
T_CMB = 2.725

BETA_D = 1.54  # d0 : 1.54     ACT : 1.5


def g1(nu_GHz: int) -> float:
    """
    Calculate g1 as a function of frequency (in GHz) and CMB temperature.

    Parameters:
    nu_GHz (float): Frequency in GHz
    TCMB (float): CMB temperature in Kelvin

    Returns:
    float: Calculated value of g1
    """
    # Convert frequency from GHz to Hz
    nu_Hz = nu_GHz * 1e9

    # Compute x = h * nu / (k_B * TCMB)
    x = (h * nu_Hz) / (k_B * T_CMB)

    # Calculate g1
    g1_val = ((x**2 * np.exp(x)) / ((np.exp(x) - 1) ** 2)) ** -1
    return g1_val


def B(nu_GHz: int, T: float) -> float:
    """
    Calculate g1 as a function of frequency (in GHz) and CMB temperature.

    Parameters:
    nu_GHz (float): Frequency in GHz
    TCMB (float): CMB temperature in Kelvin

    Returns:
    float: Calculated value of g1
    """
    # Convert frequency from GHz to Hz
    nu_Hz = nu_GHz * 1e9

    # Compute x = h * nu / (k_B * TCMB)
    x = (h * nu_Hz) / (k_B * T)

    # Calculate g1
    B_val = (2 * h * nu_Hz**3) / (c**2 * (np.exp(x) - 1))
    return B_val


def D_l_ratio(nu_1, nu_2):
    D_l_1 = (nu_1 ** (BETA_D - 2) * B(nu_1, T_D)) ** 2 * g1(nu_1) ** 2
    D_l_2 = (nu_2 ** (BETA_D - 2) * B(nu_2, T_D)) ** 2 * g1(nu_2) ** 2
    return D_l_1 / D_l_2


ALPHA_DUST = -2.42  # -2.42
# ALPHA_DUST = -2.5  # -2.42
NU_353 = 364.2

def F_dust(nu_GHz: int, beta=BETA_D) -> float:
    B_nu = B(nu_GHz, T_D)
    B_353 = B(NU_353, T_D)
    F_d_nu = ((nu_GHz / NU_353) ** (beta - 2) * B_nu / B_353) ** 2
    return F_d_nu

def frequency_scaling(nu_GHz, beta=BETA_D) -> float:
    return F_dust(nu_GHz, beta=beta) * (g1(nu_GHz) ** 2)

def multipole_scaling(ls, type='Cls', alpha=None) -> float:
    alpha = alpha or ALPHA_DUST
    if type == 'Cls':
        fac = ls * (ls + 1) / (2 * pi)
    elif type == 'Dls':
        fac = ls * 0 + 1
    return ((ls / 500) ** (alpha + 2)) / fac

def D_l_dust(
    ls: list,
    nu_GHz: int,
    a_dust: float,
    beta: float = BETA_D,
    alpha: float = ALPHA_DUST,
    return_ls: bool = False,
) -> np.ndarray:
    D_l_dust_nu = (
        a_dust * multipole_scaling(ls, type='Dls', alpha=alpha) * frequency_scaling(nu_GHz, beta=beta)
    )
    if return_ls:
        return ls, D_l_dust_nu
    else:
        return D_l_dust_nu

BETA_S = -3
ALPHA_S = -3
NU_S = 28.4

def frequency_scaling_synchrotron(nu_GHz, beta=BETA_S) -> float:
    return (nu_GHz / NU_S) ** (2 * beta)  * (g1(nu_GHz) ** 2)

def multipole_scaling_synchrotron(ls, type='Cls', alpha=ALPHA_S) -> float:
    if type == 'Cls':
        fac = ls * 0 + 1
    elif type == 'Dls':
        fac = ls * (ls + 1) / (2 * pi)
    return ((ls / 80) ** (alpha)) * fac

def D_l_synchrotron(
    ls: list,
    nu_GHz: int,
    a_synchrotron: float,
    beta: float = BETA_S,
    return_ls: bool = False,
    alpha: float = ALPHA_S,
) -> np.ndarray:
    D_l_sync_nu = (
        a_synchrotron * multipole_scaling_synchrotron(ls, type='Dls', alpha=alpha) * frequency_scaling_synchrotron(nu_GHz, beta=beta)
    )
    if return_ls:
        return ls, D_l_sync_nu
    else:
        return D_l_sync_nu


def dust_fg_Cls(
    ls: list, nu_GHz: list, wanted_keys=["EE", "BB"], amp_dust=None, beta=None, r_fg=None, alpha=None
):
    if amp_dust is None:
        amp_EE = 6.
    else:
        amp_EE = amp_dust
    alpha = alpha or ALPHA_DUST
    r_fg = r_fg or 3.5 / 6
    amp_BB = amp_EE * r_fg
    beta = beta or BETA_D
    Cls = {}
    Cls["EE"] = (
        D_l_dust(ls, nu_GHz, amp_EE, alpha=alpha, beta=beta)
        * 2
        * 3.1415
        / (ls * (ls + 1))
    )
    Cls["BB"] = (
        D_l_dust(ls, nu_GHz, amp_BB, alpha=alpha, beta=beta)
        * 2
        * 3.1415
        / (ls * (ls + 1))
    )
    if ls[0] == 1:
        Cls["EE"][0] = 0.0
        Cls["BB"][0] = 0.0
    for missing_key in [key for key in wanted_keys if key not in Cls.keys()]:
        Cls[missing_key] = np.zeros_like(Cls[list(Cls.keys())[0]], dtype=float)
    Cls = {key: Cls[key] for key in wanted_keys}
    return Cls


def sync_fg_Cls(ls: list[int], nu_GHz: float, wanted_keys=["EE", "BB"], amp=None, beta=None, r_E_B = None, alpha=None):
    if amp is None:
        amp_EE = 0.018
    else:
        amp_EE = amp
    r_E_B = r_E_B or 0.25
    alpha = alpha or ALPHA_S
    beta = beta or BETA_S
    amp_BB = amp_EE * r_E_B
    
    fac = ls * (ls + 1) / (2 * pi)

    Cls = {}
    Cls["EE"] = (
        D_l_synchrotron(ls, nu_GHz, a_synchrotron=amp_EE, beta=beta, alpha=alpha)
        / fac
    )
    Cls["BB"] = (
        D_l_synchrotron(ls, nu_GHz, a_synchrotron=amp_BB, beta=beta, alpha=alpha)  # , alpha_d=-2.54
        / fac
    )
    if ls[0] == 1:
        Cls["EE"][0] = 0.0
        Cls["BB"][0] = 0.0
    for missing_key in [key for key in wanted_keys if key not in Cls.keys()]:
        Cls[missing_key] = np.zeros_like(Cls[list(Cls.keys())[0]], dtype=float)
    Cls = {key: Cls[key] for key in wanted_keys}
    # Cls = {key: Cls[key] * (np.exp(- ((ls- 80)/50) ** 2) / 2 + 1) for key in wanted_keys}

    return Cls



def compute_rescale_fac(
    nu_GHz_in: int,
    nu_GHz_out: int,
) -> float:
    """Rescale factor assuming power law dust emission.
    Square it for spectra rescaling

    Args:
        nu_GHz_in (int): _description_
        nu_GHz_out (int): _description_

    Returns:
        float: rescale factor
    """
    return np.sqrt(F_dust(nu_GHz_out) * (g1(nu_GHz_out)**2) / F_dust(nu_GHz_in) / (g1(nu_GHz_in) **2))


def rescale_spectra(
    data: dict[list],
    nu_GHz_in: int,
    nu_GHz_out: int,
) -> dict[list]:
    """Rescale foregrounds spectra assuming a power law dust emission

    Args:
        data (dict[list]): Cls or Dls
        nu_GHz_in (int): _description_
        nu_GHz_out (int): _description_

    Returns:
        dict[list]: _description_
    """
    rescale_fac = compute_rescale_fac(nu_GHz_in, nu_GHz_out) ** 2
    data_out = {key: data[key] * rescale_fac for key in data.keys()}
    return data_out



bin_size = 10
def fit_power_law(ls: np.ndarray, Cls: np.ndarray, nu_GHz: float, spec: str = None):
    spec = spec or 'EE'
    def wrapper_power_law(ls: np.ndarray, amp_dust:float):
        # print(dust_fg_Cls(ls, nu_GHz=FREQ, amp_dust=amp_dust)[spec])
        return dust_fg_Cls(ls, nu_GHz=nu_GHz, amp_dust=amp_dust)[spec]
    bestfit, cov = curve_fit(
        wrapper_power_law, 
        bin_array(ls, bin_size),
        bin_array(Cls, bin_size),
        sigma=np.absolute(bin_array(Cls, bin_size) / np.sqrt(bin_size) / np.sqrt(2 * bin_array(ls, bin_size) + 1)),
        # p0=10
        # absolute_sigma=True
    )
    return bestfit[0], cov[0, 0]