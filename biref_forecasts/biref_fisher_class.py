import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from math import pi
from pspy import so_spectra
from biref_forecasts.biref_full_likelihood_construction import (
    SplitsSpectraCombinations,
    all_full_rotation_matrix_devs,
    full_rotation_matrix,
    get_cov_full_cls,
)
from biref_forecasts import utils, foregrounds
from functions.simu import mk_foregrounds_Cls

spectra_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
LCDM_FILENAME = "data/spectra/LCDM_spectra.txt"
class FisherBiref:
    def __init__(
        self,
        splits: list = ["s0"],
        lmin: int = 100,
        lmax: int = 1000,
        bin_size: int = 1,
        fsky: float = 1,
        combination_method: str = "comb_w_auto_EB",
        foregrounds: list[list[str]] = None,
        nu_GHzs: list[int] = None,
        amp_dust: list[float] = [None],
        amp_sync: list[float] = None,
        r_fg_dust: float = None,
        r_fg_sync: float = None,
        alphas_mapping: list[int] = None,
        bls_filenames: list[str] = None,
        noise_filenames: dict[str] = None,
        Nls_filenames: dict[str] = None,
        Nls_factor: float = None,
    ):
        """Class to do fisher forecasts of birefringence.

        Args:
            splits (list, optional): Splits/maps used. Defaults to ["s0"].
            lmin (int, optional): Minimum multipole. Defaults to 100.
            lmax (int, optional): Maximum multipole. Defaults to 1000.
            bin_size (int, optional): Bin width for constant binning. Defaults to 1.
            fsky (float, optional): Sky fraction for covariance computation. Defaults to 1.
            combination_method (str, optional): Method used to choose which auto/cross spectra
            to use. Defaults to "comb_w_auto_EB".
            foregrounds (list[list[str]], optional): List of the list of all the foregrounds for each split. 
            'power_law' for dust power law and 'power_law_sync'. Ex : [['power_law', 'powr_law_sync'], ...]
            nu_GHzs (list[int], optional): List of frequency of each splits. Defaults to None.
            amp_dust (list[float], optional): Amplitudes of the dust used in "power_law" foreground preset. Defaults to [None].
            amp_sync (list[float], optional): Amplitudes of the sync used in "power_law" foreground preset. Defaults to [None].
            r_fg_dust (float, optional): Ratio between dust BB and EE. Defaults to None.
            r_fg_sync (float, optional): Ratio between sync BB and EE. Defaults to None.
            alphas_mapping (list[int], optional): Mapping of the miscalibration angles to the splits. Defaults to None.
            bls_filenames (list[str], optional): beam filenames (.txt with first column ls and second bls). Defaults to None.
            noise_filenames (dict[str], optional): noise filenames (so_spectra .dat). Defaults to None.
            Nls_filenames (dict[str], optional): Nls filenames (so_spectra .dat) Overrides bls_fn and noise_fn. Defaults to None.
            Nls_factor (float, optional): Multiply all Nls by a given factor (Useful to cut splits into subsplits with twice the noise). 
            Defaults to False.
        """

        self.splits = splits
        self.N_splits = len(splits)
        self.SSComb_likelihood = SplitsSpectraCombinations(
            splits, method=combination_method
        )
        self.splits_spec_likelihood = self.SSComb_likelihood.get_splits_spec_list()
        self.SSComb_product = SplitsSpectraCombinations(splits, method="product")
        self.splits_spec_product = self.SSComb_product.get_splits_spec_list()

        self.splits_mask: list[bool] = [
            split in self.splits_spec_likelihood for split in self.splits_spec_product
        ]
        
        self.ls = utils.bin_array(np.arange(lmin, lmax + 1), binning=bin_size)
        self.ls_unbinned = np.arange(0, lmax + 1)
        self.fac = utils.bin_array(
            np.arange(lmin, lmax + 1) * (np.arange(lmin, lmax + 1) + 1),
            binning=bin_size,
        ) / (2 * pi)
        self.l_mask = (self.ls_unbinned >= lmin) & (self.ls_unbinned <= lmax)
        self.lmin = lmin
        self.lmax = lmax
        self.fsky = fsky
        self.bin_size = bin_size
        self.combination_method = combination_method
        self.alphas_mapping = alphas_mapping or [i for i in range(self.N_splits)]
        assert len(self.alphas_mapping) == self.N_splits
        
        # Initialize foreground kws with default values :
        self.nu_GHzs = nu_GHzs or [143 for _ in self.splits]
        self.amp_dust = amp_dust or [6.5 for _ in amp_dust]
        self.amp_sync = amp_sync or [a / 6.5 * 0.017 for a in amp_dust]
        self.r_fg_dust = r_fg_dust or 0.5
        self.r_fg_sync = r_fg_sync or 0.2


        # If no Nls, compute it with noise and beam
        if Nls_filenames is None:
            # If no beam, bls = 1
            if bls_filenames is None:
                self.bls = {
                    split: np.ones_like(self.ls, dtype=float)
                    for split in self.splits
                }
            else:
                self.bls = {
                    split: utils.bin_array(
                        np.loadtxt(filename).T[1][lmin : lmax + 1], binning=bin_size
                    )
                    for split, filename in zip(self.splits, bls_filenames)
                }
            # If no noise, assume no noise
            if noise_filenames is None:
                self.Nls = {key: np.zeros_like(self.ls, dtype=float) for key in self.splits_spec_product}
            else:
                self.Nls_per_split = {
                    split: so_spectra.read_ps(filename, spectra=spectra_pspy)[1]
                    for split, filename in zip(self.splits, noise_filenames)
                }
                self.Nls = {
                    key: utils.bin_array(self.Nls_per_split[splits[0]][spec][lmin : lmax + 1], bin_size) 
                    / self.fac
                    / self.bls[splits[0]]
                    / self.bls[splits[1]]
                    for key, splits, spec in zip(
                        self.splits_spec_product,
                        self.SSComb_product.get_splits_comb(),
                        self.SSComb_product.get_spectra(),
                    ) if splits[0] == splits[1]
                }
        else:
            # Reads the Nls files (has to be in Dls)
            self.Nls_per_split = {
                split: so_spectra.read_ps(filename, spectra=spectra_pspy)[1]
                for split, filename in zip(self.splits, Nls_filenames)
            }
            self.Nls = {
                key: utils.bin_array(self.Nls_per_split[splits[0]][spec][lmin : lmax + 1], bin_size) 
                / self.fac
                for key, splits, spec in zip(
                    self.splits_spec_product,
                    self.SSComb_product.get_splits_comb(),
                    self.SSComb_product.get_spectra(),
                ) if splits[0] == splits[1]
            }
        # Multiply Nls by a given factor if necessary
        if Nls_factor is not None:
            self.Nls = {key: nls * Nls_factor for key, nls in self.Nls.items()}

        ### FIDUCIAL LCDM : read and bin
        ls_LCDM, self.fiduc_Cls = so_spectra.read_ps('data/spectra/LCDM_spectra.txt', spectra=spectra_pspy)
        l_mask_LCDM = (ls_LCDM >= lmin) & (ls_LCDM <= lmax)
        self.fiduc_Cls = {
            key: utils.bin_array(self.fiduc_Cls[key][:lmax + 1][self.l_mask], bin_size)
            for key in self.fiduc_Cls.keys()
        }
        
        ### FOREGROUND SPECTRA
        
        # If no foregrounds kw given, assume only dust
        foregrounds = foregrounds or [["power_law"] for _ in self.splits]
        self.foregrounds = foregrounds
        self.fg_Cls = np.zeros((self.N_splits), dtype=dict)
        self.fg_Cls_dict = {}
        for i in range(self.N_splits):
            _, fg_cls = mk_foregrounds_Cls(
                self.lmax,
                nu_GHz=nu_GHzs[i],
                presets=foregrounds[i],
                amp_dust=self.amp_dust[i],
                amp_sync=self.amp_sync[i],
                r_E_B_dust=self.r_fg_dust,
                r_E_B_sync=self.r_fg_sync
            )
            self.fg_Cls[i] = {
                key: utils.bin_array(fg_cls[key][self.l_mask], self.bin_size)
                for key in fg_cls.keys()
            }
            self.fg_Cls_dict[self.splits[i]] = self.fg_Cls[i]

        self.data_fg = {
            key: np.sqrt(
                self.fg_Cls_dict[self.SSComb_product.get_splits_comb()[i][0]][
                    self.SSComb_product.get_spectra()[i]
                ]
                * self.fg_Cls_dict[self.SSComb_product.get_splits_comb()[i][1]][
                    self.SSComb_product.get_spectra()[i]
                ]
            )
            for i, key in enumerate(self.splits_spec_product)
        }

    def fisher(self, beta=0., alphas=None, summed=True, psi=None) -> np.ndarray[float]:
        if alphas is None:
            alphas = np.full(len(set(self.alphas_mapping)), 0.0, dtype=float)
        else:
            assert len(alphas) == len(set(self.alphas_mapping))
            alphas = np.array(alphas)

        angles = np.array([beta, *alphas])
        N_params = len(angles)
        if psi is not None:
            N_params += 4
        
        # Compute Cls derivatives
        full_R_CMB_devs = all_full_rotation_matrix_devs(
            self.splits,
            beta,
            alphas,
            method_for_combinations="product",
            alphas_mapping=self.alphas_mapping,
        )
        full_R_fg_devs = all_full_rotation_matrix_devs(
            self.splits,
            0.0,
            alphas,
            method_for_combinations="product",
            alphas_mapping=self.alphas_mapping,
        )
        full_R_fg_devs[0] *= 0.0

        Cls_1dev = np.zeros(
            (N_params, len(self.splits_spec_likelihood), len(self.ls))
        )
        for i in range(len(angles)):
            Cls_1dev[i] = (
                np.dot(
                    full_R_CMB_devs[i],
                    [self.fiduc_Cls[key] for key in self.SSComb_product.get_spectra()],
                )[self.splits_mask]
                + np.dot(
                    full_R_fg_devs[i],
                    np.array([self.data_fg[key] for key in self.splits_spec_product]),
                )[self.splits_mask]
            )
        if psi is not None:
            bins_A_ell = [
                (50, 130),
                (131, 210),
                (211, 510),
                (511, 1490)
            ]
            
            EE_splits = [f'E{s1}E{s2}' for s1, s2 in self.SSComb_likelihood.get_splits_comb()]
            bins_A_ell_masks = [(self.ls >= lmin_bin) & (self.ls <= lmax_bin) for lmin_bin, lmax_bin in bins_A_ell]
            
            dev_EB_dust = np.array([self.ls**-1 * self.data_fg[key][50] for key in EE_splits]) * np.sin(4 * psi)

            dev_EB_dust_binned = np.zeros((4, len(self.splits_spec_likelihood), len(self.ls)))
            for m, mask_ell in enumerate(bins_A_ell_masks):
                dev_EB_dust_binned[m][:, mask_ell] = dev_EB_dust[:, mask_ell]
                Cls_1dev[len(angles)+m] = dev_EB_dust_binned[m]

        
        # Compute Cls for cov (not needed if small angles)
        
        # full_R_for_cov = full_rotation_matrix_zeros(
        #     self.splits, method_for_combinations="product"
        # )
        # full_R_beta_for_cov = full_rotation_matrix_zeros(
        #     self.splits, method_for_combinations="product"
        # )
        # data_CMB_list = np.dot(
        #     full_R_beta_for_cov,
        #     [self.fiduc_Cls[key] for key in self.SSComb_product.get_spectra()],
        # )
        # data_fg_list = np.dot(
        #     full_R_for_cov,
        #     [self.data_fg[key] for key in self.splits_spec_product],
        # )
        data_CMB_list = np.array([self.fiduc_Cls[key] for key in self.SSComb_product.get_spectra()])
        data_fg_list = np.array([self.data_fg[key] for key in self.splits_spec_product])
        data_list = data_CMB_list + data_fg_list

        # Add noise to autospectra
        for s, spec in enumerate(self.splits_spec_product):
            if self.SSComb_product.get_noisy()[s]:
                data_list[s] += self.Nls[spec]

        # Cov matrix
        data_dict = {
            key: data_list[i] for i, key in enumerate(self.splits_spec_product)
        }
        cov_fisher = get_cov_full_cls(
            self.ls,
            data_dict,
            self.splits_spec_likelihood,
            self.fsky,
            self.bin_size,
        )
        
        cov_inv = np.linalg.inv(cov_fisher.transpose(2, 0, 1))  # shape: (N_bins, N_cls, N_cls)
        cov_inv = cov_inv.transpose(1, 2, 0)  # back to (N_cls, N_cls, N_bins)

        # i, j : params
        # b : \ell bins
        # a , c : cls, summed over
        fisher_matrix = np.einsum(
            'iab,acb,jcb->ijb',
            Cls_1dev, cov_inv, Cls_1dev, dtype=float
        )
        if summed:
            fisher_matrix_summed = np.sum(fisher_matrix, axis=2)
            return fisher_matrix_summed
        else:
            return fisher_matrix
    
    
    
    def analytical_fisher(
        self, beta: float = 0, alphas=None, summed=True, parametrization='b,a', fg_mapping=None  # parametrization is 'b,a' or 'a+b,a'
    ) -> np.ndarray[float]:
        
        EE = self.fiduc_Cls['EE']
        BB = self.fiduc_Cls['BB']
        ls = self.ls
        cov_fac = self.fsky * (2 * ls + 1)
        N = np.array([utils.bin_array(self.Nls_per_split[split]['EE'][self.lmin : self.lmax + 1], self.bin_size) for split in self.splits]) / self.fac

        if fg_mapping is None:
            r = self.r_fg_dust
            f = foregrounds.multipole_scaling(ls, type="Cls")
            g = np.sqrt([foregrounds.frequency_scaling(nu_GHz) for nu_GHz in self.nu_GHzs])
            A_E = self.amp_dust[0] * f
            N_comb = 1 / np.sum(1 / N.T, axis=1)
            g_comb = np.sum(g / N.T, axis=1) * N_comb
            sigma = np.sqrt(np.sum(g**2 / N.T, axis=1) * N_comb - g_comb**2)
        else:
            r = [self.r_fg_dust if key == 'dust' else self.r_fg_sync if key == 'sync' else 0 for key in fg_mapping]
            f_dict = {
                'dust': foregrounds.multipole_scaling(ls, type="Cls"),
                'sync': foregrounds.multipole_scaling_synchrotron(ls, type="Cls"),
            }
            amps_dict = {
                'dust': self.amp_dust[0],
                'sync': self.amp_sync[0]
            }
            freq_scaling_dict = {
                'dust': [foregrounds.frequency_scaling(nu_GHz) for nu_GHz in self.nu_GHzs],
                'sync': [foregrounds.frequency_scaling_synchrotron(nu_GHz) for nu_GHz in self.nu_GHzs],
            }
            # g = np.sqrt([self.amp_dust[0] * f * foregrounds.frequency_scaling(nu_GHz) for nu_GHz in self.nu_GHzs])
            g = np.sqrt([amps_dict[key] * f_dict[key] * freq_scaling_dict[key][i] for i, key in enumerate(fg_mapping)])

            A_E = 1.
            N_comb = 1 / np.sum(1 / N.T, axis=1)
            g_comb = np.sum(g / N, axis=0) * N_comb
            sigma = np.sqrt(np.sum(g**2 / N, axis=0) * N_comb - g_comb**2)

        eta_E = 1 / (1 + A_E * sigma**2 / N_comb)
        eta_B = 1 / (1 + r * A_E * sigma**2 / N_comb)
        VarEB = (N_comb + A_E * g_comb**2 * eta_E + EE) * (N_comb + r * A_E * g_comb**2 * eta_B + BB)

        F_bb = 4 * (cov_fac * (EE - BB) ** 2 / VarEB)
        F_fgb = 4 * (cov_fac * (EE - BB) * (1 - r) * A_E * g_comb**2 * eta_E * eta_B / VarEB)
        F_fgfg = 4 * (cov_fac * (1 - r) ** 2 * A_E**2
                            * (
                                g_comb**2 + sigma**2 * (1 + EE / N_comb)
                            )
                            / (
                                A_E * g_comb**2 + A_E * sigma**2 * (1 + EE / N_comb) + EE + N_comb
                            )
                            * (
                                g_comb**2 + sigma**2 * (1 + BB / N_comb)
                            )
                            / (
                                r * A_E * g_comb**2 + r * A_E * sigma**2 * (1 + BB / N_comb) + BB + N_comb
                            )
        )

        fisher_matrix = np.zeros((2, 2, len(self.ls)), dtype=float)
        if parametrization == 'b,a':
            fisher_matrix[0, 0] = F_bb
            fisher_matrix[0, 1] = F_bb + F_fgb
            fisher_matrix[1, 0] = F_bb + F_fgb
            fisher_matrix[1, 1] = F_bb + 2 * F_fgb + F_fgfg
        elif parametrization == 'a+b,a':
            fisher_matrix[0, 0] = F_bb
            fisher_matrix[0, 1] = F_fgb
            fisher_matrix[1, 0] = F_fgb
            fisher_matrix[1, 1] = F_fgfg
        else:
            AssertionError('Wrong parametrization name')
        fisher_matrix *= self.bin_size
        if summed:
            return np.sum(fisher_matrix, axis=2)
        else:
            return fisher_matrix
        
    def get_analytical_terms(self):
        EE = self.fiduc_Cls['EE']
        BB = self.fiduc_Cls['BB']
        ls = self.ls
        cov_fac = self.fsky * (2 * ls + 1)
        N = np.array([utils.bin_array(self.Nls_per_split[split]['EE'][self.lmin : self.lmax + 1], self.bin_size) for split in self.splits]) / self.fac

        r = self.r_fg_dust
        f = foregrounds.multipole_scaling(ls, type="Cls")
        g = np.sqrt([foregrounds.frequency_scaling(nu_GHz) for nu_GHz in self.nu_GHzs])
        A_E = self.amp_dust[0] * f
        N_comb = 1 / np.sum(1 / N.T, axis=1)
        g_comb = np.sum(g / N.T, axis=1) * N_comb
        sigma = np.sqrt(np.sum(g**2 / N.T, axis=1) * N_comb - g_comb**2)
        
        return N_comb, A_E**(1/2) * g_comb, A_E * sigma**2

    def analytical_fisher_tests(
        self, beta: float = 0, alphas=None, summed=True, parametrization='b,a'  # parametrization is 'b,a' or 'a+b,a'
    ) -> np.ndarray[float]:
        
        EE = self.fiduc_Cls['EE']
        BB = self.fiduc_Cls['BB']
        ls = self.ls
        cov_fac = self.fsky * (2 * ls + 1)
        N = np.array([utils.bin_array(self.Nls_per_split[split]['EE'][self.lmin : self.lmax + 1], self.bin_size) for split in self.splits]) / self.fac
        
        r_dust = self.r_fg_dust
        f_dust = foregrounds.multipole_scaling(ls, type="Cls")
        d = np.sqrt([foregrounds.frequency_scaling(nu_GHz) for nu_GHz in self.nu_GHzs])
        A_E_dust = self.amp_dust[0] * f_dust
        
        r_sync = self.r_fg_sync
        f_sync = foregrounds.multipole_scaling_synchrotron(ls, type="Cls")
        s = np.sqrt([foregrounds.frequency_scaling_synchrotron(nu_GHz) for nu_GHz in self.nu_GHzs])
        A_E_sync = self.amp_sync[0] * f_sync
        
        N_comb = 1 / np.sum(1 / N.T, axis=1)
        d_comb = np.sum(d / N.T, axis=1) * N_comb
        sigma_dust = np.sqrt(np.sum(d**2 / N.T, axis=1) * N_comb - d_comb**2)
        s_comb = np.sum(s / N.T, axis=1) * N_comb
        sigma_sync = np.sqrt(np.sum(s**2 / N.T, axis=1) * N_comb - s_comb**2)

        eta_E_dust = 1 / (1 + A_E_dust * sigma_dust**2 / N_comb)
        eta_B_dust = 1 / (1 + r_dust  * A_E_dust * sigma_dust**2 / N_comb)
        eta_E_sync = 1 / (1 + A_E_sync * sigma_sync**2 / N_comb)
        eta_B_sync = 1 / (1 + r_sync  * A_E_sync * sigma_sync**2 / N_comb)
        VarEB = (
            N_comb + A_E_dust * d_comb**2 * eta_E_dust + A_E_sync * s_comb**2 * eta_E_sync + EE
                 ) * (N_comb + r_dust * A_E_dust * d_comb**2 * eta_B_dust + r_sync * A_E_sync * s_comb**2 * eta_B_sync + BB)

        sig_E_CMB = EE - BB
        sig_B_CMB = EE - BB
        noi_E_CMB = N_comb + eta_E_dust * A_E_dust * d_comb**2 + A_E_sync * s_comb**2 * eta_B_sync + EE
        noi_B_CMB = N_comb + eta_B_dust * r_dust  * A_E_dust * d_comb**2 + r_sync * A_E_sync * s_comb**2 * eta_B_sync + BB

        g_eff_E_dust = A_E_dust * (d_comb**2 + sigma_dust**2 * (1 + EE / N_comb))
        g_eff_B_dust = A_E_dust * (d_comb**2 + sigma_dust**2 * (1 + BB / N_comb))
        sig_E_dust = (1 - r_dust ) * g_eff_E_dust
        sig_B_dust = (1 - r_dust ) * g_eff_B_dust
        noi_E_dust = g_eff_E_dust + EE + N_comb
        noi_B_dust = r_dust  * g_eff_B_dust + BB + N_comb
        
        g_eff_E_sync = A_E_sync * (s_comb**2 + sigma_sync**2 * (1 + EE / N_comb))
        g_eff_B_sync = A_E_sync * (s_comb**2 + sigma_sync**2 * (1 + BB / N_comb))
        sig_E_sync = (1 - r_sync ) * g_eff_E_sync
        sig_B_sync = (1 - r_sync ) * g_eff_B_sync
        noi_E_sync = g_eff_E_sync + EE + N_comb
        noi_B_sync = r_sync  * g_eff_B_sync + BB + N_comb
        
        sig_E_fg = sig_E_dust + sig_E_sync
        sig_B_fg = sig_B_dust + sig_B_sync
        noi_E_fg = noi_E_dust + noi_E_sync
        noi_B_fg = noi_B_dust + noi_B_sync
        
        F_bb = 4 * cov_fac * sig_E_CMB * sig_B_CMB / (noi_E_CMB * noi_B_CMB)
        F_fgb = 4 * (cov_fac * (EE - BB) * (1 - r_sync) * A_E_sync * d_comb**2 * eta_E_sync * eta_B_sync / VarEB) * 0.
        F_fgfg = 4 * cov_fac * sig_E_fg * sig_B_fg / (noi_E_fg * noi_B_fg)

        fisher_matrix = np.zeros((2, 2, len(self.ls)), dtype=float)
        if parametrization == 'b,a':
            fisher_matrix[0, 0] = F_bb
            fisher_matrix[0, 1] = F_bb + F_fgb
            fisher_matrix[1, 0] = F_bb + F_fgb
            fisher_matrix[1, 1] = F_bb + 2 * F_fgb + F_fgfg
        elif parametrization == 'a+b,a':
            fisher_matrix[0, 0] = F_bb
            fisher_matrix[0, 1] = F_fgb
            fisher_matrix[1, 0] = F_fgb
            fisher_matrix[1, 1] = F_fgfg
        else:
            AssertionError('Wrong parametrization name')
        fisher_matrix *= self.bin_size
        if summed:
            return np.sum(fisher_matrix, axis=2)
        else:
            return fisher_matrix

    def bias(
        self, gamma=1e-2, bias_type=1,
    ) -> dict[np.ndarray[float]]:
        """USE THIS WITH ONLY ONE MISCALIBRATION ANGLE

        Args:
            gamma (_type_, optional): _description_. Defaults to 1e-2.
            bias_type (int, optional): 0: CMB, 1:FG. Defaults to 1.

        Returns:
            dict[np.ndarray[float]]: _description_
        """
        
        fisher_matrix_inv = np.linalg.inv(self.analytical_fisher(0., parametrization='a+b,a'))
        fisher_matrix_l = self.analytical_fisher(0., summed=False, parametrization='a+b,a')
        self.biases_array = fisher_matrix_inv @ np.transpose(fisher_matrix_l[:, bias_type, :], (0, 1)) * gamma
        self.biases = {angle: self.biases_array[a] for a, angle in enumerate(['a+b', 'a'])}
        self.biases['b'] = self.biases['a+b'] - self.biases['a']
        return self.biases
    
    def plot(
        self, beta=0., alphas=None, wanted_splits_comb=None, type="Cl", log=False
    ):
        if alphas is None:
            alphas = np.full_like(self.splits, 0.0, dtype=float)
        else:
            alphas = np.array(alphas)

        if wanted_splits_comb is None:
            wanted_splits_comb = [f'E{self.splits[0]}B{self.splits[1]}']
        for comb in wanted_splits_comb:
            assert comb in self.splits_spec_product 

        wanted_index = int(
            np.where(np.array(self.splits_spec_product) == wanted_splits_comb[0])[0]
        )
        if type == "Cl":
            fac = self.ls * 0 + 1
        elif type == "lCl":
            fac = self.ls
        elif type == "Dl":
            fac = self.ls * (self.ls + 1)
        elif type == "l2Dl":
            fac = self.ls * (self.ls + 1) * self.ls**2
        else:
            raise KeyError(f'{type} not in [Cl, lCl, Dl, l2Dl]')

        # Compute Cls for cov
        spectra_list_for_cov = self.SSComb_product.get_spectra()
        full_R_for_cov = full_rotation_matrix(
            self.splits, 0.0, alphas, method_for_combinations="product"
        )
        full_R_beta_for_cov = full_rotation_matrix(
            self.splits, beta, alphas, method_for_combinations="product"
        )
        data_CMB_list = np.dot(
            full_R_beta_for_cov,
            [self.fiduc_Cls[key] for key in spectra_list_for_cov],
        )
        data_fg_list = np.dot(
            full_R_for_cov,
            [self.data_fg[key] for key in self.splits_spec_product],
        )
        data_list = data_CMB_list + data_fg_list

        # Add noise to autospectra
        for s, spec in enumerate(self.splits_spec_product):
            if spec in self.Nls.keys():
                data_list[s] += self.Nls[spec]

        # Cov matrix
        data_dict = {
            key: data_list[i] for i, key in enumerate(self.splits_spec_product)
        }
        cov_fisher = get_cov_full_cls(
            self.ls,
            data_dict,
            self.splits_spec_product,
            self.fsky,
            self.bin_size,
        )

        # fac = self.ls * (self.ls + 1)
        fig, axs = plt.subplots(
            1, 1, figsize=(8, 5), dpi=100,
        )
        if log:
            axs.set_yscale("log")
        for i, key in enumerate(wanted_splits_comb):
            axs.plot(
                self.ls,
                (data_fg_list[wanted_index] + data_CMB_list[wanted_index]) * fac,
                label=f"{key} LCDM+fg",
                color="purple",
                alpha=0.7,
            )
            axs.plot(
                self.ls,
                data_CMB_list[wanted_index] * fac,
                label=f"{key} LCDM",
                color="c",
            )
            axs.plot(
                self.ls,
                data_fg_list[wanted_index] * fac,
                label=f"{key} fg",
                color="red",
            )
            axs.set_ylabel(type)
            ylims = axs.get_ylim()
            axs.errorbar(
                self.ls,
                data_dict[key] * fac,
                np.sqrt(cov_fisher[wanted_index, wanted_index]) * fac,
                label=f"{key} Obs",
                marker=".",
                linestyle="none",
                color="black",
            )
            axs.set_ylim(ylims)
            axs.legend()
        return axs

