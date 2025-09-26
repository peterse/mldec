import stim
import numpy as np
from mldec.utils import bit_tools
from itertools import chain, combinations
import itertools
from functools import reduce


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def operator_sequence_to_stim(lst, n):
    if len(lst) > 0:
        out = reduce(lambda P, Q: P*Q, lst)
    else:
        out = stim.PauliString("_" * n) 
    return out


def num_pauli_to_str(idx, n, paulitype):
    """
    Convert a list of indices to a stim-friendly pauli string.

    Example:
        num_pauli_to_str([1, 4, 5], 7, "X") -> "_X__XX_"

    Args:
        idx: list of indices
        n: total number of qubits
        paulitype: "X" or "Z" or "Y"
    """
    pauli = ["_"] * n
    for i in idx:
        pauli[i] = paulitype
    return "".join(pauli)


def build_lst_lookup(n, generators_S, generators_T, generators_L, Hx, Hz, cache=True, css=True):
    """a lookup table that assigns each error a (sigma, logical) coset label

    the rows are indexed by binary(error), each column is concatenated [sigma, logical]

    returns:
         size 4**n array of concatenated [sigma, logical] labels, indexed by binary(error)
    """
        
    keys = np.zeros((4**n, 2*n), dtype=int).reshape(2**(n-1), 4, 2**(n-1), -1)
    vals = np.zeros((4**n, n+1), dtype=int).reshape(2**(n-1), 4, 2**(n-1), -1)

    out = np.zeros((4 ** n, 2 ** (n+1)), dtype=bool)
    for i_sigma, pure_error_lst in enumerate(powerset(generators_T)):
        pure_error = operator_sequence_to_stim(pure_error_lst, n)
        if css:
            xerr, zerr = pure_error.to_numpy()
            sigma_z = (Hz @ xerr) % 2
            sigma_x = (Hx @ zerr) % 2
            sigma = np.concatenate((sigma_x, sigma_z), axis=0)
        else:
            H = np.concatenate((Hx, Hz), axis=1)
            perr = pure_error.to_numpy()
            perr = np.array([perr[0], perr[1]]).flatten()
            sigma = (H @ perr) % 2
        logical_masks = np.array(list(itertools.product([0, 1], repeat=2)))
        for j_logical, logical_mask in enumerate(logical_masks):
            logical_op_lst = [generators_L[i] for i, bit in enumerate(logical_mask) if bit]
            logical_op = operator_sequence_to_stim(logical_op_lst, n)            
            for k_stab, stabilizer_lst in enumerate(powerset(generators_S)):
                stabilizer = operator_sequence_to_stim(stabilizer_lst, n)
                error = pure_error * logical_op * stabilizer
                error_symplectic = np.concatenate(error.to_numpy(), axis=0).astype(int)
                # out[bits_to_ints(error_symplectic)[0]] = np.concatenate((sigma, logical_mask), axis=0)
                keys[i_sigma, j_logical, k_stab] = error_symplectic
                vals[i_sigma, j_logical, k_stab] = np.concatenate((sigma, logical_mask), axis=0)
    vals = vals.reshape(-1, n+1)
    out = vals[np.argsort(bit_tools.bits_to_ints(keys.reshape(-1, 2*n)))]

    return out


def build_syndrome_probs_and_weight_distr(generators_S, generators_T, generators_L, Hx, Hz, noise_model):
    """Compute a table of syndrome probabilities, and a table of error weight distribution by syndrome.

    Returns:
        p_SL: a dictionary mapping (sigma, ell) to the probability of that pair
        hist_SL_wts: a dictionary mapping (sigma, ell) an (n+1) array of weights, where the ith entry is the 
            cum. probability of errors in that coset enumerator having weight i
    """

    p_SL = {}
    hist_SL_wts = {}

    r = len(generators_S)
    n = len(generators_S[0])
    k = n - r
    assert len(generators_L) == 2*k
    assert len(generators_T) == r
    for a, pure_error_lst in enumerate(powerset(generators_T)):

        pure_error = operator_sequence_to_stim(pure_error_lst, n)
        if css:
            xerr, zerr = pure_error.to_numpy()
            sigma_z = (Hz @ xerr) % 2
            sigma_x = (Hx @ zerr) % 2
            sigma = np.concatenate((sigma_x, sigma_z), axis=0)
        else:
            H = np.concatenate((Hx, Hz), axis=1)
            perr = pure_error.to_numpy()
            perr = np.array([perr[0], perr[1]]).flatten()
            sigma = (H @ perr) % 2
        p_sigma = 0
        logical_masks = np.array(list(itertools.product([0, 1], repeat=2*k)))
        for b, logical_mask in enumerate(logical_masks):
            # to enumerate over the logicals, we want to preserve their order 
            # as provided in generators_L, as this will determine the binary
            # mask representing that logical
            p_SL[(tuple(sigma), tuple(logical_mask))] = 0
            hist_SL_wts[(tuple(sigma), tuple(logical_mask))] = np.zeros(n+1)
            logical_op_lst = [generators_L[i] for i, bit in enumerate(logical_mask) if bit]
            logical_op = operator_sequence_to_stim(logical_op_lst, n)
            for c, stabilizer_lst in enumerate(powerset(generators_S)):
                stabilizer = operator_sequence_to_stim(stabilizer_lst, n)
                error = pure_error * logical_op * stabilizer
                error_symplectic = np.concatenate(error.to_numpy(), axis=0).astype(int)
                p_err = noise_model(error_symplectic, n)
                wt = sum(np.logical_or(error_symplectic[:n], error_symplectic[n:]))

                # increment the probability of this (sigma, ell) pair, and the weight enumerator for this
                p_SL[(tuple(sigma), tuple(logical_mask))] += p_err
                hist_SL_wts[(tuple(sigma), tuple(logical_mask))][wt] += p_err
            

    return p_SL, hist_SL_wts