import numpy as np
import stim

import os

import itertools
from mldec.codes import code_utils
from mldec.utils import bit_tools

# get abspath of the directory containing this module
abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
CACHE = os.path.join(abspath, "cache")


def fivequbit_code_stabilizers(n):
    """
   
    """
    pauli_stabilizers = [
        "XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"

    ]
    H_x = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
    ])
    H_z = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ])
    stabilizers = [stim.PauliString(stabilizer) for stabilizer in pauli_stabilizers]
    return stabilizers, H_x, H_z


def fivequbit_code_logicals(n):
    """
    The logical operator is X...X either horizontally or vertically across the lattice, in the unrotated version.
    In the rotated version 
    """

    x_L = stim.PauliString("XXXXX")
    z_L = stim.PauliString("ZZZZZ")
    return x_L, z_L


def build_lst_lookup(n, cache=True):
    """a lookup table that assigns each error a (sigma, logical) coset label

    the rows are indexed by binary(error), each column is concatenated [sigma, logical]

    returns:
         size 4**n array of concatenated [sigma, logical] labels, indexed by binary(error)
    """
    assert n == 5
    if cache:
        target = f"fivequbit_code/n{n}_LST.npy"
        path = os.path.join(CACHE, target)
        if os.path.exists(path):
            out = np.load(path)
            if out is not None:
                return out

    generators_S, generators_T, generators_L, Hx, Hz = generators_STL_Hx_Hz(generators, generators_L)
    out = code_utils.build_lst_lookup(n, generators_S, generators_T, generators_L, Hx, Hz, cache)
    
    if cache:
        np.save(path, out)
    return out


def generators_STL_Hx_Hz(n):
    """Create the generators for the stabilizer gp, pure error gp, and logical gp, along with pcms."""
    generators, Hx, Hz = fivequbit_code_stabilizers(n)

    t = stim.Tableau.from_stabilizers(generators, allow_redundant=True, allow_underconstrained=True)
    # we leave out the final stabilizer that canonically represents the degree of 
    # freedom for a state we didn't specify
    generators_S = [t.z_output(k) for k in range(len(t) - 1)] # stabilizer generators, ordered X type then Z type
    generators_T = [t.x_output(k) for k in range(len(t) - 1)] # pure error generators, arbitrary order
    generators_L = fivequbit_code_logicals(n)
    return generators_S, generators_T, generators_L, Hx, Hz