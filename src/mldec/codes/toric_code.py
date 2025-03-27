import numpy as np
import stim

import os

import itertools
from mldec.codes import toric_code, code_utils
from mldec.utils import bit_tools

# get abspath of the directory containing this module
abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
CACHE = os.path.join(abspath, "cache")

def index_vertex(x, y, L):
    """Indexing scheme for qubits on lattice."""
    return x + y * L


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


def rotated_surface_code_stabilizers(L):
    """
    construct the set of stabilizers for the rotated toric code having periodic boundary conditions.

    Tiling numbering scheme, e.g. L=3
            Z
         0 --- 1 --- 2
         |  X  |  Z  | X
         3 --- 4 --- 5
       X |  Z  |  X  |        <--- THIS IS A QUANTUM COMPUTER. 
         6 --- 7 --- 8
                  Z
    Returns:
        x_stabilizers: list of r_x total stim.PauliString objects representing the X stabilizers
        z_stabilizers: list of r_z total stim.PauliString objects representing the Z stabilizers
        H_x: PCM for x stabilizers, (r_x, n) matrix
        H_z: PCM for z stabilizers, (r_z, n) matrix
    """
    x_stabilizers = []
    z_stabilizers = []
    n = L**2  # total number of qubits in the standard toric code layout

    # For each vertex (x, y), define a star operator: product of X on the 4 edges touching that vertex.
    # Edges: (x,y) horiz, (x-1,y) horiz, (x,y) vert, (x,y-1) vert, all mod L for periodic BCs.

    for x in range(L):
        for y in range(L):
            v1 = index_vertex(x, y, L)
            if x == L - 1 or y == L - 1:
                continue
            v2 = index_vertex(x, (y + 1), L)
            v3 = index_vertex((x + 1), y, L)
            v4 = index_vertex((x + 1), (y + 1), L)
            vertices = [v1, v2, v3, v4]
            if ( x + y ) % 2 == 0:
                ptype = "X"
                x_stabilizers.append(vertices)
            else:
                ptype = "Z"     
                z_stabilizers.append(vertices)

    # top and bottom
    for x in range(L):
        if x < L - 1:
            ptype = "Z"
            if x % 2 == 0 :
                v1 = index_vertex(x, 0, L)
                v2 = index_vertex(x + 1, 0, L)
                z_stabilizers.append([v1, v2])
            elif x % 2 == 1:
                v1 = index_vertex(x, L - 1, L)
                v2 = index_vertex(x + 1, L - 1, L)
                z_stabilizers.append([v1, v2])

    for y in range(L):
        if y < L - 1:
            ptype = "X"
            if y % 2 == 1:
                v1 = index_vertex(0, y, L)
                v2 = index_vertex(0, y + 1, L)
                x_stabilizers.append([v1, v2])
            elif y % 2 == 0:
                v1 = index_vertex(L - 1, y, L)
                v2 = index_vertex(L - 1, y + 1, L)
                x_stabilizers.append([v1, v2])
    H_x = np.zeros((len(x_stabilizers), n), dtype=int)
    for i, js in enumerate(x_stabilizers):
        H_x[i, js] = 1
    H_z = np.zeros((len(z_stabilizers), n), dtype=int)
    for i, js in enumerate(z_stabilizers):
        H_z[i, js] = 1
    z_stabilizers = [num_pauli_to_str(stabilizer, n, "Z") for stabilizer in z_stabilizers]
    z_stabilizers = [stim.PauliString(stabilizer) for stabilizer in z_stabilizers]
    x_stabilizers = [num_pauli_to_str(stabilizer, n, "X") for stabilizer in x_stabilizers]
    x_stabilizers = [stim.PauliString(stabilizer) for stabilizer in x_stabilizers]

    return x_stabilizers, z_stabilizers, H_x, H_z


def rotated_toric_code_logicals(L):
    """
    The logical operator is X...X either horizontally or vertically across the lattice, in the unrotated version.
    In the rotated version 
    """
    if L != 3:
        raise NotImplementedError("Only L=3 is implemented")
    n = L**2
    x_L = num_pauli_to_str([3, 4, 5], n, "X")
    x_L = stim.PauliString(x_L)
    z_L = num_pauli_to_str([1, 4, 7], n, "Z")
    z_L = stim.PauliString(z_L)
    return x_L, z_L

def generators_STL_Hx_Hz(L):
    """Create the generators for the stabilizer gp, pure error gp, and logical gp, along with pcms."""
    if L != 3:
        raise NotImplementedError("Only L=3 is implemented")
    x_generators, z_generators, Hx, Hz = rotated_surface_code_stabilizers(L)
    generators = x_generators + z_generators
    t = stim.Tableau.from_stabilizers(generators, allow_redundant=True, allow_underconstrained=True)
    # we leave out the final stabilizer that canonically represents the degree of 
    # freedom for a state we didn't specify
    generators_S = [t.z_output(k) for k in range(len(t) - 1)] # stabilizer generators, ordered X type then Z type
    generators_T = [t.x_output(k) for k in range(len(t) - 1)] # pure error generators, arbitrary order
    generators_L = rotated_toric_code_logicals(3) # logicals, ordered X type then Z type
    return generators_S, generators_T, generators_L, Hx, Hz



def build_lst_lookup(L, cache=True):
    """Build a lookup table labelling errors by (sigma, logical) coset labels

    the rows are indexed by binary(error), each column is concatenated [sigma, logical]

    returns:
         size 4**n array of concatenated [sigma, logical] labels, indexed by binary(error)
    """
    if L != 3:
        raise NotImplementedError("Only L=3 is implemented for now.")
    n = 9
    if cache:
        target = f"L{L}_LST.npy"
        path = os.path.join(CACHE, target)
        if os.path.exists(path):
            out = np.load(path)
            if out is not None:
                return out
        
    keys = np.zeros((4**n, 2*n), dtype=int).reshape(2**(n-1), 4, 2**(n-1), -1)
    vals = np.zeros((4**n, n+1), dtype=int).reshape(2**(n-1), 4, 2**(n-1), -1)

    generators_S, generators_T, generators_L, Hx, Hz = toric_code.generators_STL_Hx_Hz(L)
    out = np.zeros((4 ** n, 2 ** (n+1)), dtype=bool)
    for i_sigma, pure_error_lst in enumerate(code_utils.powerset(generators_T)):
        pure_error = code_utils.operator_sequence_to_stim(pure_error_lst, n)
        xerr, zerr = pure_error.to_numpy()
        sigma_z = (Hz @ xerr) % 2
        sigma_x = (Hx @ zerr) % 2
        sigma = np.concatenate((sigma_x, sigma_z), axis=0)
        logical_masks = np.array(list(itertools.product([0, 1], repeat=2)))
        for j_logical, logical_mask in enumerate(logical_masks):
            logical_op_lst = [generators_L[i] for i, bit in enumerate(logical_mask) if bit]
            logical_op = code_utils.operator_sequence_to_stim(logical_op_lst, n)            
            for k_stab, stabilizer_lst in enumerate(code_utils.powerset(generators_S)):
                stabilizer = code_utils.operator_sequence_to_stim(stabilizer_lst, n)
                error = pure_error * logical_op * stabilizer
                error_symplectic = np.concatenate(error.to_numpy(), axis=0).astype(int)
                # out[bits_to_ints(error_symplectic)[0]] = np.concatenate((sigma, logical_mask), axis=0)
                keys[i_sigma, j_logical, k_stab] = error_symplectic
                vals[i_sigma, j_logical, k_stab] = np.concatenate((sigma, logical_mask), axis=0)
    vals = vals.reshape(-1, n+1)
    out = vals[np.argsort(bit_tools.bits_to_ints(keys.reshape(-1, 2*n)))]

    if cache:
        np.save(path, out)
    return out