import os
import numpy as np
import stim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch


from mldec.utils import graph_representation




# get abspath of the directory containing this module
abspath = os.path.dirname(os.path.abspath(__file__))
EXP_DATA_DIR = os.path.join(abspath, "exp_data")

def save_data(X, Y, fname):
    global EXP_DATA_DIR
    torch.save(X, os.path.join(EXP_DATA_DIR, fname + "_X.pt"), _use_new_zipfile_serialization=False)
    torch.save(Y, os.path.join(EXP_DATA_DIR, fname + "_Y.pt"), _use_new_zipfile_serialization=False)


def load_data(fname):
    global EXP_DATA_DIR
    X = torch.load(os.path.join(EXP_DATA_DIR, fname + "_X.pt"), weights_only=False)
    Y = torch.load(os.path.join(EXP_DATA_DIR, fname + "_Y.pt"), weights_only=False)
    return X, Y


def make_exp_dataset_name(code_size, repetitions, beta):
    """Make a name for repetition code data.
    
    code_size: 'n', the number of data qubits in the [n, 1, n] rep code.
    repetitions: number of syndrome measurement rounds, usually NOT including final meassurement
    beta: the parameter describing error modification to the experiment

    Note that this will be called from some other experimental script. The data should respect
    the following guidelines: 
        - beta=0 indicates validation data (i.e. represents beta=1).
        - data should have shape (repetitions, n). The first axis represents RAW syndrome readouts,
            the second axis is the concatentation of [n-1] bits of syndrome, and 1 bit of I{observable flipped}
    
    The experimental data is stored in datasets/exp_data/.
    """
    return f"rep_code_data_n{code_size}_reps{repetitions}_beta{beta}"


def sample_dataset(n_data, dataset_config, device, seed=None):
    """Given a dataset config, sample an EXPERIMENTAL dataset of size n_data.
    
    The experimental data is stored

    FIXME:
    - for a big experiment, we will have many different ways to slice an [n, 1, n] rep code into [m, 1, m] with m<n
          ...where do we implement that?
    """

    repetitions = dataset_config.get("repetitions") # "cycles" of measurement
    code_size = dataset_config.get("code_size")
    beta = dataset_config.get("beta")

    # get the experimental data
    fname = make_exp_dataset_name(code_size, repetitions, beta)
    X, y = load_data(fname)
    # slicing to fit our datasize: FIXME: do this in a randomized way.
    if len(X) < n_data:
        raise ValueError(f"Not enough data to sample {n_data} samples. Only {len(X)} samples available.")
    X = X[:n_data]
    y = y[:n_data]
    # X has shape (n_data, repetitions, n-1), y has shape (n_data, repetitions,)
    non_empty_indices = (np.sum(X.reshape(n_data, -1), axis = 1) != 0)

    trivial_count = len(y[~non_empty_indices])
    syndrome_data = X[non_empty_indices, :, :]
    observable_flips = y[non_empty_indices]
    buffer = generate_batch(syndrome_data, observable_flips)
    torch_buffer = dataset_to_torch(buffer, device)
    # return signature is for backwards compatibility with a stim sampler...
    return torch_buffer, trivial_count, observable_flips


def build_syndrome_2D(syndrome_data):
    """
    Construct a 2D space/time gride of detector flips over time.
    The first row always matches the initial syndrome, and then 
    each subsequent row are differences between the previous row and the current row.
    Args:
        syndrome_data: shape (repetitions, n-1)
    Returns:
        out: shape (n-1, repetitions)
    """
    out = np.zeros_like(syndrome_data)
    out[0,:] = syndrome_data[0,:]
    out[1:,:] = (syndrome_data[1:,:] - syndrome_data[:-1,:]) % 2
    return out.transpose(1, 0)


def get_2D_graph(syndrome_2D, 
    target = None,
    m_nearest_nodes = None,
    power = None):
    """
    Form a graph from a repeated syndrome measurement where a node is added, 
    each time the syndrome changes. The node features are 2D.

    The main part of this function generates edge weights that are like 
    1/d^2 for distance d=max(x-distance, t-distance) between any two nodes (faults).

    Args:
        syndrome_2D: shape (n-1, repetitions) of detector flip events
    
    Returns:
        list of [graph nodes, graph edges, graph edge attributes, graph labels]
    """
    # get defect indices: this is a pair (x_coords, t_coords) for the faults.
    X = np.vstack(np.nonzero(syndrome_2D)).T
    
    # set default power of inverted distances to 1
    if power is None:
        power = 1.

    # construct the adjacency matrix. This is going to do inverse-square scaling in 
    # the max(x-distance, t-distance) for any pairs of faults with x-distance, t-distance between
    # their respective coordinates. This is...potentially reasonable.
    x_coord = X[:,0].reshape(-1, 1)
    t_coord = X[:,1].reshape(-1, 1)
    x_dist = np.abs(x_coord.T - x_coord) 
    t_dist = np.abs(t_coord.T - t_coord) 
    # inverse square of the supremum norm between two nodes
    Adj = np.maximum.reduce([x_dist, t_dist])
    # set diagonal elements to nonzero to circumvent division by zero
    np.fill_diagonal(Adj, 1)
    # scale the edge weights: This adjacency matrix represents
    # weights that are 'decaying' in node distance
    Adj = 1./Adj ** power
    # set diagonal elements to zero to exclude self loops
    np.fill_diagonal(Adj, 0)

    # remove all but the m_nearest neighbours
    if m_nearest_nodes is not None:
        for ix, row in enumerate(Adj.T):
            # Do not remove edges if a node has (degree <= m)
            if np.count_nonzero(row) <= m_nearest_nodes:
                continue
            # Get indices of all nodes that are not the m nearest
            # Remove these edges by setting elements to 0 in adjacency matrix
            Adj.T[ix, np.argpartition(row,-m_nearest_nodes)[:-m_nearest_nodes]] = 0.

    Adj = np.maximum(Adj, Adj.T) # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj) # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    if target is not None:
        y = target.reshape(1, 1)
    else:
        y = None

    return [X.astype(np.float32), edge_index.astype(np.int64,), edge_attr.astype(np.float32), y.astype(np.float32)]


def generate_batch(syndrome_data_list, observable_flips_list, power=2, m_nearest_nodes=None):
    '''
    Generates a batch of graphs from a list of stim experiments.
    '''
    batch = []
    for i in range(len(syndrome_data_list)):
        # convert to syndrome grid with flips instead of faults:
        syndrome_2D = build_syndrome_2D(syndrome_data_list[i])
        # get the logical equivalence class:
        true_eq_class = np.array([int(observable_flips_list[i])])
        # map to graph representation

        graph = get_2D_graph(syndrome_2D = syndrome_2D,
                            target = true_eq_class,
                            power = power,
                            m_nearest_nodes = m_nearest_nodes)
        batch.append(graph)
    return batch


def dataset_to_torch(buffer, device):
    # convert list of numpy arrays to torch Data object containing torch tensors
    batch = []
    for i in range(len(buffer)):
        buffer_i = buffer[i]
        X = torch.from_numpy(buffer_i[0]).to(device)
        edge_index = torch.from_numpy(buffer_i[1]).to(device)
        edge_attr = torch.from_numpy(buffer_i[2]).to(device)
        y = torch.from_numpy(buffer_i[3]).to(device)
        batch.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return batch