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


def sample_dataset(n_data, dataset_config, device):
    """Given a dataset config, sample a dataset of size n_data.
    
    Since a large fraction of data are trivial, we will also keep track
    of how many 'no error' events were sampled and return this, but otherwise not include such 
    data in the training set. This procedure allows you to
    calculate accuracy as

        acc = (correct_predictions_on_data + trivial_count) / n_data

    Returns:
        torch_buffer: A list of torch Data objects, each containing a graph representation 
            of the data.
        trivial_count: The number of trivial syndromes in the dataset.

    """

    repetitions = dataset_config.get("repetitions") # "cycles" of measurement
    code_size = dataset_config.get("code_size")
    p_base = dataset_config.get("p")
    beta = dataset_config.get("beta")
    p = p_base * beta
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
    detector_coordinates = detector_coordinates.astype(np.uint8)
    sampler = circuit.compile_detector_sampler()

    # get the surface code grid:
    mask = syndrome_mask(code_size, repetitions)
    stim_data, observable_flips = sampler.sample(shots=n_data, separate_observables=True)
    non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
    trivial_count = len(observable_flips[~ non_empty_indices])
    stim_data = stim_data[non_empty_indices, :]
    observable_flips = observable_flips[non_empty_indices]

    # This code will let you generate dataset with only nontrivial data...
    # factor = max(1/(20*p), 10)
    # shots = int(factor * n_data)
    # stim_data, observable_flips = [], []
    # trivial_count = 0
    # while len(stim_data) < (n_data):
    #     stim_data_it, observable_flips_it = sampler.sample(shots=shots, separate_observables=True)
    #     # remove empty syndromes:

    #     non_empty_indices = (np.sum(stim_data_it, axis = 1) != 0)
    #     new_data = stim_data_it[non_empty_indices, :]
    #     new_obs = observable_flips_it[non_empty_indices]
    #     if len(new_data) + len(new_obs) > n_data:
    #         # we now need to truncate nicely so that there are n_data, but the proportion of non-empty syndromes
    #         # correctly models the underlying distribution of trivial syndromes; the easiest way is to 
    #         # finish off sampling ineficiently
    #         shots = 1
    #         continue

    #     stim_data.extend(new_data)
    #     observable_flips.extend(new_obs)
    #     trivial_count += len(observable_flips_it[~ non_empty_indices])
    buffer = generate_batch(stim_data, observable_flips, detector_coordinates, mask)
    torch_buffer = dataset_to_torch(buffer, device)

    return torch_buffer, trivial_count


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