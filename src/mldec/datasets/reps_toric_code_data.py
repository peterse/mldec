import numpy as np
import stim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch


from mldec.utils import graph_representation

def syndrome_mask(code_size, repetitions):
    '''
    Creates a surface code grid. 1: X-stabilizer. 3: Z-stabilizer.
    '''
    M = code_size + 1
    syndrome_matrix_X = np.zeros((M, M), dtype = np.uint8)
    # starting from northern boundary:
    syndrome_matrix_X[::2, 1:M - 1:2] = 1
    # starting from first row inside the grid:
    syndrome_matrix_X[1::2, 2::2] = 1
    syndrome_matrix_Z = np.rot90(syndrome_matrix_X) * 3
    # Combine syndrome matrices where 1 entries 
    # correspond to x and 3 entries to z defects
    syndrome_matrix = (syndrome_matrix_X + syndrome_matrix_Z)
    # Return the syndrome matrix
    return np.dstack([syndrome_matrix] * (repetitions + 1))


def stim_to_syndrome_3D(mask, coordinates, stim_data):
    '''
    Converts a stim detection event array to a syndrome grid. 
    1 indicates a violated X-stabilizer, 3 a violated Z stabilizer. 
    Only the difference between two subsequent cycles is stored.
    '''
    # initialize grid:
    syndrome_3D = np.zeros_like(mask)

    # first to last time-step:
    syndrome_3D[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = stim_data

    # only store the difference in two subsequent syndromes:
    syndrome_3D[:, :, 1:] = (syndrome_3D[:, :, 1:] - syndrome_3D[:, :, 0: - 1]) % 2

    # convert X (Z) stabilizers to 1(3) entries in the matrix
    syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]

    return syndrome_3D


def generate_batch(stim_data_list,
                observable_flips_list,
                detector_coordinates,
                mask, m_nearest_nodes=None, power=2):
    '''
    Generates a batch of graphs from a list of stim experiments.
    '''
    batch = []

    for i in range(len(stim_data_list)):
        # convert to syndrome grid:
        syndrome = stim_to_syndrome_3D(mask, detector_coordinates, stim_data_list[i])
        # get the logical equivalence class:
        true_eq_class = np.array([int(observable_flips_list[i])])
        # map to graph representation
        graph = graph_representation.get_3D_graph(syndrome_3D = syndrome,
                            target = true_eq_class,
                            power = power,
                            m_nearest_nodes = m_nearest_nodes)
        batch.append(graph)
    return batch


def sample_dataset(n_data, dataset_config):

    repetitions = dataset_config.get("repetitions")
    code_size = dataset_config.get("code_size")
    p = dataset_config.get("p")
    # Initialize stim circuit for a fixed training rate
    circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds = repetitions,
                distance = code_size,
                after_clifford_depolarization = p,
                after_reset_flip_probability = p,
                before_measure_flip_probability = p,
                before_round_data_depolarization = p)
    # get detector coordinates (same for all error rates):
    detector_coordinates = circuit.get_detector_coordinates()
    # get coordinates of detectors (divide by 2 because stim labels 2d grid points)
    # coordinates are of type (d_west, d_north, hence the reversed order)
    detector_coordinates = np.array(list(detector_coordinates.values()))
    # rescale space like coordinates:
    detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
    # convert to integers
    detector_coordinates = detector_coordinates.astype(np.uint8)
    sampler = circuit.compile_detector_sampler()

    # get the surface code grid:
    mask = syndrome_mask(code_size, repetitions)
    factor = max(1/(20*p), 10)
    stim_data, observable_flips = [], []
    while len(stim_data) < (n_data):
        stim_data, observable_flips = sampler.sample(shots = factor*n_data, separate_observables=True)
        # remove empty syndromes:
        non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
        stim_data.extend(stim_data[non_empty_indices, :])
        observable_flips.extend(observable_flips[non_empty_indices])
    
    buffer = generate_batch(stim_data, observable_flips, detector_coordinates, mask)
    torch_buffer = dataset_to_torch(buffer, device)

    return torch_buffer


def generate_test_batch(test_size):
    '''Generates a test batch at one test error rate'''
    # Keep track of trivial syndromes
    correct_predictions_trivial = 0
    stim_data_list, observable_flips_list = [], []

    stim_data, observable_flips = sampler.sample(shots=test_size, separate_observables = True)
    # remove empty syndromes:
    # (don't count imperfect X(Z) in second to last time)
    non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
    stim_data_list.extend(stim_data[non_empty_indices, :])
    observable_flips_list.extend(observable_flips[non_empty_indices])
    # count empty instances as trivial predictions: 
    correct_predictions_trivial += len(observable_flips[~ non_empty_indices])
    # if there are more non-empty syndromes than necessary
    stim_data_list = stim_data_list[: test_size]
    observable_flips_list = observable_flips_list[: test_size]
    buffer = generate_batch(stim_data_list, observable_flips_list,
                            detector_coordinates, mask, m_nearest_nodes, power)
    test_batch = dataset_to_torch(buffer, device)

    return test_batch, correct_predictions_trivial



def dataset_to_torch(buffer, device):
    # convert list of numpy arrays to torch Data object containing torch GPU tensors
    batch = []
    for i in range(len(buffer)):
        X = torch.from_numpy(buffer[i][0]).to(device)
        edge_index = torch.from_numpy(buffer[i][1]).to(device)
        edge_attr = torch.from_numpy(buffer[i][2]).to(device)
        y = torch.from_numpy(buffer[i][3]).to(device)
        batch.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y = y))
    return batch