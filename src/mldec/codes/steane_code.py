import numpy as np
import stim

import os

import itertools
from mldec.codes import code_utils
from mldec.utils import bit_tools

# get abspath of the directory containing this module
abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
CACHE = os.path.join(abspath, "cache")

def index_vertex(x, y, L):
    """Indexing scheme for qubits on lattice."""
    return x + y * L



def steane_code_stabilizers(n):
    """
   
    """
    H_x = np.array([
        [0,0,0,1,1,1,1],
        [0,1,1,0,0,1,1],
        [1,0,1,0,1,0,1]
    ])
    # copy H_x
    H_z = H_x.copy()
    masks = [
        [3, 4, 5, 6],
        [1, 2, 4, 5],
        [0, 2, 4, 6]
    ]

    z_stabilizers = [code_utils.num_pauli_to_str(stabilizer, n, "Z") for stabilizer in masks]
    x_stabilizers = [code_utils.num_pauli_to_str(stabilizer, n, "X") for stabilizer in masks]
    z_stabilizers = [stim.PauliString(stabilizer) for stabilizer in z_stabilizers]
    x_stabilizers = [stim.PauliString(stabilizer) for stabilizer in x_stabilizers]

    return x_stabilizers, z_stabilizers, H_x, H_z


def steane_code_logicals(n):
    """
    The logical operator is X...X either horizontally or vertically across the lattice, in the unrotated version.
    In the rotated version 
    """

    x_L = stim.PauliString("XXXXXXX")
    z_L = stim.PauliString("ZZZZZZZ")
    return x_L, z_L


def build_lst_lookup(n, cache=True):
    """a lookup table that assigns each error a (sigma, logical) coset label

    the rows are indexed by binary(error), each column is concatenated [sigma, logical]

    returns:
         size 4**n array of concatenated [sigma, logical] labels, indexed by binary(error)
    """
    assert n == 7
    if cache:
        target = f"steane_code/n{n}_LST.npy"
        path = os.path.join(CACHE, target)
        if os.path.exists(path):
            out = np.load(path)
            if out is not None:
                return out

    generators_S, generators_T, generators_L, Hx, Hz = generators_STL_Hx_Hz(n)
    out = code_utils.build_lst_lookup(n, generators_S, generators_T, generators_L, Hx, Hz, cache)
    
    if cache:
        np.save(path, out)
    return out


def generators_STL_Hx_Hz(n):
    """Create the generators for the stabilizer gp, pure error gp, and logical gp, along with pcms."""
    x_generators, z_generators, Hx, Hz = steane_code_stabilizers(n)
    generators = x_generators + z_generators
    t = stim.Tableau.from_stabilizers(generators, allow_redundant=True, allow_underconstrained=True)
    # we leave out the final stabilizer that canonically represents the degree of 
    # freedom for a state we didn't specify
    generators_S = [t.z_output(k) for k in range(len(t) - 1)] # stabilizer generators, ordered X type then Z type
    generators_T = [t.x_output(k) for k in range(len(t) - 1)] # pure error generators, arbitrary order
    generators_L = steane_code_logicals(n)
    return generators_S, generators_T, generators_L, Hx, Hz