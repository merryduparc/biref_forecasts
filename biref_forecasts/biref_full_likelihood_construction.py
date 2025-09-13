import itertools
import numpy as np
import math
from utils import SplitsSpectraCombinations, spectra_combinations_from_splits, extract_spectra_pairs, extract_splits_pairs


def rotation_matrix(theta_i: float, theta_j: float) -> np.ndarray:
    """
    Constructs the 4x4 transformation matrix in EE BB EB BE space for given theta_i and theta_j.
    """
    cos_2ti = np.cos(2 * theta_i)
    sin_2ti = np.sin(2 * theta_i)
    cos_2tj = np.cos(2 * theta_j)
    sin_2tj = np.sin(2 * theta_j)
    return np.array(
        [
            [
                cos_2ti * cos_2tj,
                sin_2ti * sin_2tj,
                -cos_2ti * sin_2tj,
                -sin_2ti * cos_2tj,
            ],
            [
                sin_2ti * sin_2tj,
                cos_2ti * cos_2tj,
                sin_2ti * cos_2tj,
                cos_2ti * sin_2tj,
            ],
            [
                cos_2ti * sin_2tj,
                -sin_2ti * cos_2tj,
                cos_2ti * cos_2tj,
                -sin_2ti * sin_2tj,
            ],
            [
                sin_2ti * cos_2tj,
                -cos_2ti * sin_2tj,
                -sin_2ti * sin_2tj,
                cos_2ti * cos_2tj,
            ],
        ]
    )


def rotation_matrix_idev(theta_i: float, theta_j: float) -> np.ndarray:
    cos_2ti = math.cos(2 * theta_i)
    sin_2ti = math.sin(2 * theta_i)
    cos_2tj = math.cos(2 * theta_j)
    sin_2tj = math.sin(2 * theta_j)
    return np.array(
        [
            [
                -2 * sin_2ti * cos_2tj,
                2 * cos_2ti * sin_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
            ],
            [
                2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
                2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * sin_2tj,
            ],
            [
                -2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
            ],
            [
                2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
            ],
        ]
    )


def rotation_matrix_jdev(theta_i: float, theta_j: float) -> np.ndarray:
    cos_2ti = math.cos(2 * theta_i)
    sin_2ti = math.sin(2 * theta_i)
    cos_2tj = math.cos(2 * theta_j)
    sin_2tj = math.sin(2 * theta_j)
    return np.array(
        [
            [
                -2 * cos_2ti * sin_2tj,
                2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
            ],
            [
                2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * sin_2tj,
                2 * cos_2ti * cos_2tj,
            ],
            [
                2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
            ],
            [
                -2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
            ],
        ]
    )


def rotation_matrix_faster(
    cos_2ti: float, sin_2ti: float, cos_2tj: float, sin_2tj: float
) -> np.ndarray:
    """
    Constructs the 4x4 transformation matrix in EE BB EB BE space for given theta_i and theta_j.
    """
    return np.array(
        [
            [
                cos_2ti * cos_2tj,
                sin_2ti * sin_2tj,
                -cos_2ti * sin_2tj,
                -sin_2ti * cos_2tj,
            ],
            [
                sin_2ti * sin_2tj,
                cos_2ti * cos_2tj,
                sin_2ti * cos_2tj,
                cos_2ti * sin_2tj,
            ],
            [
                cos_2ti * sin_2tj,
                -sin_2ti * cos_2tj,
                cos_2ti * cos_2tj,
                -sin_2ti * sin_2tj,
            ],
            [
                sin_2ti * cos_2tj,
                -cos_2ti * sin_2tj,
                -sin_2ti * sin_2tj,
                cos_2ti * cos_2tj,
            ],
        ]
    )


def rotation_matrix_idev_faster(cos_2ti, sin_2ti, cos_2tj, sin_2tj):
    return np.array(
        [
            [
                -2 * sin_2ti * cos_2tj,
                2 * cos_2ti * sin_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
            ],
            [
                2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
                2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * sin_2tj,
            ],
            [
                -2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
            ],
            [
                2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
            ],
        ]
    )


def rotation_matrix_jdev_faster(cos_2ti, sin_2ti, cos_2tj, sin_2tj):
    return np.array(
        [
            [
                -2 * cos_2ti * sin_2tj,
                2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
            ],
            [
                2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * sin_2tj,
                2 * cos_2ti * cos_2tj,
            ],
            [
                2 * cos_2ti * cos_2tj,
                2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * sin_2tj,
                -2 * sin_2ti * cos_2tj,
            ],
            [
                -2 * sin_2ti * sin_2tj,
                -2 * cos_2ti * cos_2tj,
                -2 * sin_2ti * cos_2tj,
                -2 * cos_2ti * sin_2tj,
            ],
        ]
    )


def rotation_matrix_bothdev(theta_i, theta_j):
    return rotation_matrix_idev(theta_i, theta_j) + rotation_matrix_jdev(
        theta_i, theta_j
    )


def rotation_matrix_bothdev_faster(cos_2ti, sin_2ti, cos_2tj, sin_2tj):
    return rotation_matrix_idev_faster(
        cos_2ti, sin_2ti, cos_2tj, sin_2tj
    ) + rotation_matrix_jdev_faster(cos_2ti, sin_2ti, cos_2tj, sin_2tj)


def rotation_matrix_element(
    theta_i, theta_j, initial_state, target_state, which_matrix="rot"
):
    """
    Given initial (X_i Y_j) and target (W_i Z_j), returns the matrix element that
    transforms (X_i Y_j) into (W_i Z_j)_rotated.

    Possible states for `initial_state` and `target_state` are:
        'EE', 'BB', 'EB', 'BE' (first letter is always _i and last is _j)
    """
    # Map states to matrix indices
    state_to_index = {"EE": 0, "BB": 1, "EB": 2, "BE": 3}

    # Validate input states
    if initial_state not in state_to_index or target_state not in state_to_index:
        raise ValueError("Invalid state. Choose from 'EE', 'BB', 'EB', 'BE'")

    # Get the matrix element
    if which_matrix == "rot":
        matrix = rotation_matrix(theta_i, theta_j)
    elif which_matrix == "rot_idev":
        matrix = rotation_matrix_idev(theta_i, theta_j)
    elif which_matrix == "rot_jdev":
        matrix = rotation_matrix_jdev(theta_i, theta_j)
    elif which_matrix == "rot_bothdev":
        matrix = rotation_matrix_bothdev(theta_i, theta_j)
    row = state_to_index[target_state]
    col = state_to_index[initial_state]

    return matrix[row, col]


def full_rotation_matrix(
    splits, beta, alphas, method_for_combinations="comb", which_matrix="rot"
):
    assert len(splits) == len(alphas)
    alphas_dict = {split: alphas[i] for i, split in enumerate(splits)}
    splits_combinations_list = spectra_combinations_from_splits(
        splits, method=method_for_combinations
    )
    spectra_list = extract_spectra_pairs(splits_combinations_list)
    splits_pairs_list = extract_splits_pairs(splits_combinations_list)
    full_R = np.zeros((len(splits_combinations_list), len(splits_combinations_list)))

    # if comb = 'Es0Bs1', then spectrum = 'EB' and splits_pair = ['s0', 's1']
    for i, (comb_i, spectrum_i, splits_pair_i) in enumerate(
        zip(splits_combinations_list, spectra_list, splits_pairs_list)
    ):
        for j, (comb_j, spectrum_j, splits_pair_j) in enumerate(
            zip(splits_combinations_list, spectra_list, splits_pairs_list)
        ):
            if splits_pair_i == splits_pair_j:
                full_R[j, i] = rotation_matrix_element(
                    alphas_dict[splits_pair_i[0]] + beta,
                    alphas_dict[splits_pair_i[1]] + beta,
                    spectrum_i,
                    spectrum_j,
                    which_matrix=which_matrix,
                )
    return full_R


def full_rotation_matrix_faster(
    splits, beta, alphas, method_for_combinations="comb", which_matrix="rot"
):
    assert len(splits) == len(alphas)
    alphas = np.array(alphas)
    cos_2alphasbeta = np.cos(2 * (alphas + beta))
    sin_2alphasbeta = np.sin(2 * (alphas + beta))
    alphas_dict = {split: alphas[i] for i, split in enumerate(splits)}
    splits_spec_comb = SplitsSpectraCombinations(splits, method=method_for_combinations)
    splits_combinations_list = splits_spec_comb.get_splits_spec_list()
    spectra_list = splits_spec_comb.get_spectra()
    splits_pairs_list = splits_spec_comb.get_splits_comb()
    full_R = np.zeros((len(splits_combinations_list), len(splits_combinations_list)))

    rot_matrices = {
        alpha_name_1: {
            alpha_name_2: np.zeros((4, 4)) for alpha_name_2 in alphas_dict.keys()
        }
        for alpha_name_1 in alphas_dict.keys()
    }
    for i, (alpha_name_1, alpha_1) in enumerate(alphas_dict.items()):
        for j, (alpha_name_2, alpha_2) in enumerate(alphas_dict.items()):
            if which_matrix == "rot":
                rot_matrices[alpha_name_1][alpha_name_2] = rotation_matrix_faster(
                    cos_2alphasbeta[i],
                    sin_2alphasbeta[i],
                    cos_2alphasbeta[j],
                    sin_2alphasbeta[j],
                )
            elif which_matrix == "rot_idev":
                rot_matrices[alpha_name_1][alpha_name_2] = rotation_matrix_idev_faster(
                    cos_2alphasbeta[i],
                    sin_2alphasbeta[i],
                    cos_2alphasbeta[j],
                    sin_2alphasbeta[j],
                )
            elif which_matrix == "rot_jdev":
                rot_matrices[alpha_name_1][alpha_name_2] = rotation_matrix_jdev_faster(
                    cos_2alphasbeta[i],
                    sin_2alphasbeta[i],
                    cos_2alphasbeta[j],
                    sin_2alphasbeta[j],
                )
            elif which_matrix == "rot_bothdev":
                rot_matrices[alpha_name_1][alpha_name_2] = (
                    rotation_matrix_bothdev_faster(
                        cos_2alphasbeta[i],
                        sin_2alphasbeta[i],
                        cos_2alphasbeta[j],
                        sin_2alphasbeta[j],
                    )
                )

    # Cache the rotation matrices
    rot_cache = {
        (key1, key2): matrix
        for key1, sub_dict in rot_matrices.items()
        for key2, matrix in sub_dict.items()
    }
    spec2index = {"EE": 0, "BB": 1, "EB": 2, "BE": 3}
    # if comb = 'Es0Bs1', then spectrum = 'EB' and splits_pair = ['s0', 's1']
    for i, (spectrum_i, splits_pair_i) in enumerate(
        zip(spectra_list, splits_pairs_list)
    ):
        for j, (spectrum_j, splits_pair_j) in enumerate(
            zip(spectra_list, splits_pairs_list)
        ):
            if splits_pair_i == splits_pair_j:
                row = spec2index[spectrum_j]
                col = spec2index[spectrum_i]
                full_R[j, i] = rot_cache[(splits_pair_i[0], splits_pair_i[1])][row, col]
    return full_R


def full_rotation_matrix_zeros(
    splits, method_for_combinations="comb", which_matrix="rot"
):
    splits_spec_comb = SplitsSpectraCombinations(splits, method=method_for_combinations)
    splits_combinations_list = splits_spec_comb.get_splits_spec_list()
    spectra_list = splits_spec_comb.get_spectra()
    splits_pairs_list = splits_spec_comb.get_splits_comb()
    full_R = np.zeros((len(splits_combinations_list), len(splits_combinations_list)))
    rot_matrix = np.zeros((4, 4))
    if which_matrix == "rot":
        rot_matrix = rotation_matrix_faster(
            1.0,
            0.0,
            1.0,
            0.0,
        )
    elif which_matrix == "rot_idev":
        rot_matrix = rotation_matrix_idev_faster(
            1.0,
            0.0,
            1.0,
            0.0,
        )
    elif which_matrix == "rot_jdev":
        rot_matrix = rotation_matrix_jdev_faster(
            1.0,
            0.0,
            1.0,
            0.0,
        )
    elif which_matrix == "rot_bothdev":
        rot_matrix = rotation_matrix_bothdev_faster(
            1.0,
            0.0,
            1.0,
            0.0,
        )
    spec2index = {"EE": 0, "BB": 1, "EB": 2, "BE": 3}
    # if comb = 'Es0Bs1', then spectrum = 'EB' and splits_pair = ['s0', 's1']
    for i, (spectrum_i, splits_pair_i) in enumerate(
        zip(spectra_list, splits_pairs_list)
    ):
        for j, (spectrum_j, splits_pair_j) in enumerate(
            zip(spectra_list, splits_pairs_list)
        ):
            if splits_pair_i == splits_pair_j:
                row = spec2index[spectrum_j]
                col = spec2index[spectrum_i]
                full_R[j, i] = rot_matrix[row, col]
    return full_R


def full_rotation_matrix_zeros_new(
    splits: list[str],
    method_for_combinations: str = "product",
    which_matrix: str = "rot",
) -> np.ndarray:
    """Compute the matrix that rotates/derivates a data vector of Cls given by method_for_combinations.
    Advised to keep method_for_combinations='product' and then cut the data vector
    Suppose all angles = 0 for now

    Args:
        splits (list[str]): _description_
        method_for_combinations (str, optional): _description_. Defaults to "product".
        which_matrix (str, optional): rotation, derivative compared to 'left', 'right' or both splits. Defaults to "rot".

    Returns:
        np.ndarray: _description_
    """
    allowed_matrices = ["rot", "rot_idev", "rot_jdev", "rot_bothdev"]
    if which_matrix not in allowed_matrices:
        print(f"/!\ which_matrix not in {allowed_matrices}. Default to 'rot'")
        which_matrix = "rot"
    splits_spec_comb = SplitsSpectraCombinations(splits, method=method_for_combinations)
    splits_combinations_list = splits_spec_comb.get_splits_spec_list()
    spectra_nest = splits_spec_comb.get_spectra_nest()
    splits_pairs_list = splits_spec_comb.get_splits_comb()
    full_R = np.zeros((len(splits_combinations_list), len(splits_combinations_list)))
    if which_matrix == "rot":
        spec2comp_i = {"EE": 1.0, "BB": 1.0, "EB": 0.0, "BE": 0.0}
        spec2comp_j = {"EE": 1.0, "BB": 1.0, "EB": 0.0, "BE": 0.0}
    elif which_matrix == "rot_idev":
        spec2comp_i = {"EE": 0.0, "BB": 0.0, "EB": -2.0, "BE": 2.0}
        spec2comp_j = {"EE": 1.0, "BB": 1.0, "EB": 0.0, "BE": 0.0}
    elif which_matrix == "rot_jdev":
        spec2comp_i = {"EE": 1.0, "BB": 1.0, "EB": 0.0, "BE": 0.0}
        spec2comp_j = {"EE": 0.0, "BB": 0.0, "EB": -2.0, "BE": 2.0}
    # if comb = 'Es0Bs1', then spectrum = 'EB' and splits_pair = ['s0', 's1']
    if which_matrix == "rot_bothdev":
        full_R = full_rotation_matrix_zeros_new(
            splits,
            method_for_combinations=method_for_combinations,
            which_matrix="rot_idev",
        ) + full_rotation_matrix_zeros_new(
            splits,
            method_for_combinations=method_for_combinations,
            which_matrix="rot_jdev",
        )
    else:
        for idx, (spectrum_i, splits_pair_i) in enumerate(
            zip(spectra_nest, splits_pairs_list)
        ):
            # Extract precomputed indices where splits_pair_i matches
            matching_indices = [
                j for j, pair in enumerate(splits_pairs_list) if pair == splits_pair_i
            ]

            for j in matching_indices:
                spectrum_j = spectra_nest[j]
                key_i = spectrum_i[0] + spectrum_j[0]
                key_j = spectrum_i[1] + spectrum_j[1]
                full_R[idx, j] = spec2comp_i[key_i] * spec2comp_j[key_j]
    return full_R


def all_full_rotation_matrix_devs(
    splits: list[str],
    beta: float = 0.0,
    alphas: float = None,
    method_for_combinations: str = "product",
    alphas_mapping: list[int] = None,
) -> np.ndarray:
    """Compute the derivation matrices for beta and the misscalibration angles

    Args:
        splits (list[str]): _description_
        beta (float): _description_
        method_for_combinations (str, optional): keep product then cut. Defaults to "product".
        alphas_mapping (list[int], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: first axis is the angles (beta, alpha_1...)
    """
    if alphas_mapping is None:
        alphas_mapping = np.array([i for i in range(len(alphas))])
        
    assert len(alphas_mapping) == len(splits)

    angles = np.array([beta, *alphas])

    splits_spec_comb = SplitsSpectraCombinations(splits, method=method_for_combinations)
    splits_combinations_list = splits_spec_comb.get_splits_spec_list()
    splits_pairs_list = splits_spec_comb.get_splits_comb()
    full_R_dev = np.zeros(
        (len(angles), len(splits_combinations_list), len(splits_combinations_list))
    )
    if (angles == 0.).all():
        rot_matrix_idev = full_rotation_matrix_zeros_new(
            splits,
            which_matrix="rot_idev",
            method_for_combinations=method_for_combinations,
        )
        rot_matrix_jdev = full_rotation_matrix_zeros_new(
            splits,
            which_matrix="rot_jdev",
            method_for_combinations=method_for_combinations,
        )
        full_R_dev[0] += full_rotation_matrix_zeros_new(
            splits,
            which_matrix="rot_bothdev",
            method_for_combinations=method_for_combinations,
        )
    else:
        rot_matrix_idev = full_rotation_matrix(
            splits,
            beta,
            alphas,
            which_matrix="rot_idev",
            method_for_combinations=method_for_combinations,
        )
        rot_matrix_jdev = full_rotation_matrix(
            splits,
            beta,
            alphas,
            which_matrix="rot_jdev",
            method_for_combinations=method_for_combinations,
        )
        full_R_dev[0] += full_rotation_matrix(
            splits,
            beta,
            alphas,
            which_matrix="rot_bothdev",
            method_for_combinations=method_for_combinations,
        )
    for i, i_map in enumerate(alphas_mapping):
        current_split = splits[i]
        matching_first_indices = [
            comb_1_index
            for comb_1_index, pair in enumerate(splits_pairs_list)
            if current_split == pair[0]
        ]
        matching_second_indices = [
            comb_1_index
            for comb_1_index, pair in enumerate(splits_pairs_list)
            if current_split == pair[1]
        ]

        for comb_1_index in matching_first_indices:
            full_R_dev[i_map + 1][comb_1_index, :] += rot_matrix_idev[comb_1_index, :]

        for comb_1_index in matching_second_indices:
            full_R_dev[i_map + 1][comb_1_index, :] += rot_matrix_jdev[comb_1_index, :]
    return full_R_dev


def get_spectra_covariance(
    ls: np.ndarray,
    data: dict,
    X: str,
    Y: str,
    W: str,
    Z: str,
    fsky=1,
    bin_size=1,
    ignore_keys=None,
):
    """Compute :
        Cov(Cls^{X, Y}, Cls^{W, Z}) =
        1/(2l+1) (Cls^{X, W} Cls^{Y, Z}
        + Cls^{X, Z} Cls^{Y, W})
        For X, Y, W and Z in (T, E, B)

    Args:
        ls (_type_): _description_
        data (_type_): dict where keys are combinaitions of X, Y, W and Z (XY...)
        X (_type_): _description_
        Y (_type_): _description_
        W (_type_): _description_
        Z (_type_): _description_

    Returns:
        cov: the covariance array
    """
    cov = np.zeros_like(ls, dtype=float)
    nu = (2 * ls + 1) * fsky * bin_size

    if ignore_keys == None:
        cov += 1 / nu * (data[X + W] * data[Y + Z] + data[X + Z] * data[Y + W])
    else:
        if (
            X[0] + W[0] not in ignore_keys
            and W[0] + X[0] not in ignore_keys
            and Y[0] + Z[0] not in ignore_keys
            and Z[0] + Y[0] not in ignore_keys
        ):
            cov += 1 / nu * (data[X + W] * data[Y + Z])
        if (
            X[0] + Z[0] not in ignore_keys
            and Z[0] + X[0] not in ignore_keys
            and Y[0] + W[0] not in ignore_keys
            and W[0] + Y[0] not in ignore_keys
        ):
            cov += 1 / nu * (data[X + Z] * data[Y + W])
    return cov


def get_cov_full_cls(ls, data, splits_list_full, fsky=1, bin_size=1):
    """Return the cov matrices with a shape of (splits_list_full, splits_list_full, ls)

    Args:
        ls (_type_): _description_
        data (_type_): _description_
        splits_list_full (_type_): _description_
        fsky (int, optional): _description_. Defaults to 1.
        bin_size (int, optional): _description_. Defaults to 1.
    """

    cov_full_cls = np.zeros((len(splits_list_full), len(splits_list_full), len(ls)))
    for i, comb_i in enumerate(splits_list_full):
        for j, comb_j in enumerate(splits_list_full):
            cov_full_cls[i, j] = get_spectra_covariance(
                ls,
                data,
                comb_i[: len(comb_i) // 2],
                comb_i[len(comb_i) // 2 :],
                comb_j[: len(comb_j) // 2],
                comb_j[len(comb_j) // 2 :],
                fsky=fsky,
                bin_size=bin_size,
                # ignore_keys=["EB"],
            )

    return cov_full_cls
