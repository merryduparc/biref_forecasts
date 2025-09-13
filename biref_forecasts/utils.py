import numpy as np
import itertools

def bin_array(array, binning: int = None):
    """Bin a given array.
        If the array size is not a multiple of the binning,
        ignore the rest of the array.

    Args:
        array (_type_): _description_
        binning (int): _description_

    Returns:
        np.ndarray: _description_
    """
    binning = binning or 1
    array = array[: len(array) // binning * binning]
    x_reshaped = np.reshape(array, (-1, binning))
    return np.mean(x_reshaped, axis=1)


def spectra_combinations_from_splits(
    input_strings: list[str], method="comb"
) -> list[str]:
    """
    Return combinations of spectra with different methods
    comb : EE BB EB BE for i != j
    comb_w_auto : comb + EE BB EB for i == j
    product : everything EE BB EB BE for all i, j
        (contains redundancies, not recommanded for likelihood)

    Args:
        input_strings (_type_): list of str corresponding to individual splits
            (e.g. ['s0', 's1', ...])

    Returns:
        _type_: list of str with all combinations.
    """

    result = []

    if method == "comb":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}E{b}", f"B{a}B{b}", f"E{a}B{b}", f"B{a}E{b}"])
            # else:
            #     result.extend([f"E{a}E{b}", f"B{a}B{b}", "E{a}B{b}"])
    elif method == "comb_w_auto":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}E{b}", f"B{a}B{b}", f"E{a}B{b}", f"B{a}E{b}"])
            else:
                result.extend([f"E{a}E{b}", f"B{a}B{b}", f"E{a}B{b}"])
    elif method == "comb_w_auto_EB":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}E{b}", f"B{a}B{b}", f"E{a}B{b}", f"B{a}E{b}"])
            else:
                result.extend([f"E{a}B{b}"])
    elif method == "only_EB":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}B{b}", f"B{a}E{b}"])
            else:
                result.extend([f"E{a}B{b}"])
    elif method == "only_EB_auto":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a == b:
                result.extend([f"E{a}B{b}"])
    elif method == "only_EB_cross":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}B{b}", f"B{a}E{b}"])
    elif method == "product":
        iterator = itertools.product(input_strings, repeat=2)
        for a, b in iterator:
            result.extend([f"E{a}E{b}", f"B{a}B{b}", f"E{a}B{b}", f"B{a}E{b}"])
    elif method == "all_auto":
        iterator = itertools.product(input_strings, repeat=2)
        for a, b in iterator:
            result.extend([f"E{a}E{b}", f"B{a}B{b}"])
    elif method == "safe_auto":
        iterator = itertools.combinations_with_replacement(input_strings, r=2)
        for a, b in iterator:
            if a != b:
                result.extend([f"E{a}E{b}", f"B{a}B{b}"])

    return result


def extract_pairs(input_list: list[str]) -> list[list[str]]:
    """
    From a list of str in the format 'Xsplit1Ysplit2',
    returns a list of save size with elements '[Xsplit1, Ysplit2], ...'

    Args:
        input_list (_type_): input list

    Returns:
        _type_: list of the same size as input (elements are 2-long lists)
    """
    suffix_pairs = [
        [item[: len(item) // 2], item[len(item) // 2 :]] for item in input_list
    ]
    return suffix_pairs


def extract_spectra_pairs(input_list: list[str]) -> list[str]:
    """
    From a list of str in the format 'Xsplit1Ysplit2',
    returns a list of same size with only 'XY'

    Args:
        input_list (_type_): input list

    Returns:
        _type_: list of the same size as input
    """

    prefix_pairs = []
    prefix_pairs = [item[0] + item[len(item) // 2] for item in input_list]

    return prefix_pairs


def extract_spectra_pairs_nested(input_list: list[str]) -> list[str]:
    """
    From a list of str in the format 'Xsplit1Ysplit2',
    returns a list of elements ['X', 'Y']

    Args:
        input_list (_type_): input list

    Returns:
        _type_: list of the same size as input
    """
    prefix_pairs = []
    prefix_pairs = [[item[0], item[len(item) // 2]] for item in input_list]

    return prefix_pairs


def extract_splits_pairs(input_list: list[str]) -> list[list[str]]:
    """
    From a list of str in the format 'Xsplit1Ysplit2',
    returns a list of save size with elements '[split1, split2], ...'

    Args:
        input_list (_type_): input list

    Returns:
        _type_: list of the same size as input (elements are 2-long lists)
    """
    suffix_pairs = [
        [item[1 : len(item) // 2], item[len(item) // 2 + 1 :]] for item in input_list
    ]
    return suffix_pairs


def extract_unwanted_indices(input_list: list[str]) -> list[bool]:
    """Return True for indices whit : ExEx BxBx or BxEx

    Args:
        input_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    splits_pairs = extract_splits_pairs(input_list)
    spectra_pairs = extract_spectra_pairs(input_list)
    unwanted_indices = np.zeros_like(input_list, dtype=bool)
    for i in range(len(input_list)):
        if (splits_pairs[i][0] == splits_pairs[i][1]) and (
            spectra_pairs[i][0] == spectra_pairs[i][1]
        ):
            unwanted_indices[i] = True
        if (
            (splits_pairs[i][0] == splits_pairs[i][1])
            and (spectra_pairs[i][0] == "B")
            and (spectra_pairs[i][1] == "E")
        ):
            unwanted_indices[i] = True

    return unwanted_indices


def extract_noisy_indices(input_list: list[str]) -> list[bool]:
    """Same as extract_unwanted_indices but only ExEx and BxBx

    Args:
        input_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    splits_pairs = extract_splits_pairs(input_list)
    spectra_pairs = extract_spectra_pairs(input_list)
    noisy_indices = np.zeros_like(input_list, dtype=bool)
    for i in range(len(input_list)):
        if (splits_pairs[i][0] == splits_pairs[i][1]) and (
            spectra_pairs[i][0] == spectra_pairs[i][1]
        ):
            noisy_indices[i] = True

    return noisy_indices


class SplitsSpectraCombinations:
    def __init__(
        self,
        input_splits: list[str] = None,
        method: str = "comb_w_auto_EB",
    ):  
        """Silly class to quickly compute all kind of combinations of a list of bands

        Args:
            input_splits (list[str], optional): List of bands, should all have same lenght. Defaults to None.
            method (str, optional): Combination method, check spectra_combinations_for_splits. Defaults to "comb_w_auto_EB".
        """
        input_splits = input_splits or ["s0", "s1"] # Should all have the asame lenght
        self.splits_spec_list = spectra_combinations_from_splits(
            input_splits, method=method
        )
        self.pairs = None
        self.spectra = None
        self.spectra_nest = None
        self.splits_comb = None
        self.unwanted = None
        self.noisy = None

    def get_splits_spec_list(self) -> list[str]:
        return self.splits_spec_list

    def get_pairs(self) -> list[list[str]]:
        if self.pairs is None:
            self.pairs = extract_pairs(self.splits_spec_list)
        return self.pairs

    def get_spectra(self) -> list[str]:
        if self.spectra is None:
            self.spectra = extract_spectra_pairs(self.splits_spec_list)
        return self.spectra

    def get_spectra_nest(self) -> list[str]:
        if self.spectra_nest is None:
            self.spectra_nest = extract_spectra_pairs_nested(self.splits_spec_list)
        return self.spectra_nest

    def get_splits_comb(self) -> list[list[str]]:
        if self.splits_comb is None:
            self.splits_comb = extract_splits_pairs(self.splits_spec_list)
        return self.splits_comb

    def get_unwanted(self) -> list[bool]:
        if self.unwanted is None:
            self.unwanted = extract_unwanted_indices(self.splits_spec_list)
        return self.unwanted

    def get_noisy(self) -> list[bool]:
        if self.noisy is None:
            self.noisy = extract_noisy_indices(self.splits_spec_list)
        return self.noisy
