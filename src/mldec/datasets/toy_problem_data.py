"""toy_problem_data.py - This module constructs the toy problem dataset"""

import numpy as np
import torch
from mldec.utils.bit_tools import binarr


def repetition_pcm(n):
    """generate a PCM for a repetition code on n bits."""
    out = []
    for i in range(n-1):
        out.append([0]*i + [1, 1] + [0]*(n-i-2))
    return np.array(out, dtype=np.uint8)


def create_dataset(n):
    """Construct a dataset of every bitstring (errors) paired with its syndromes

    Returns:
        X: (2**n, n-k) array of syndromes
        Y: (2**n, n) array of all length-n bitstrings
    """
    H = repetition_pcm(n)
    all_bitstrings = binarr(n)
    all_syndromes = all_bitstrings @ H.T % 2
    X, Y = (all_syndromes, all_bitstrings)
    Xtensor = torch.tensor(X, dtype=torch.float32)
    Ytensor = torch.tensor(Y.copy(), dtype=torch.float32)
    return Xtensor, Ytensor


def noise_model(s, n, dataset_config, permute=None):
    """Create the biased bitflip noise model 
    
    The first n//2 bits have prob. p1 of flipping, the last n//2 have prob. p2.
    
    Warning: if the difference in bias is too much, the weight-ordering of bitstrings 
    is no longer the same as likelihood ordering. Manually check that there is no
    weight-(k+1) bitstring with higher likelihood than a weight-k bitstring!

    Args:
        s: (n_data, n) array of bitstrings.
        p1, p2: bitflip probabilities for the first and second half of the bit

    """
    p1, p2 = dataset_config['p1'], dataset_config['p2']
    if permute is not None:
        s = s[:,permute]
    p_first = torch.prod(p1*s[:,:n//2] +(1-p1)*(1-s[:,:n//2]), axis=1)
    p_second = torch.prod(p2*s[:,n//2:] + (1-p2)*(1-s[:,n//2:]), axis=1)
    return torch.multiply(p_first, p_second)


# def sample_bitstring_v1(n, p1, p2, n_data):
#     """Sample bitstrings from the biased bitflip model v1.
    
#     Args:
#         n: number of bits
#         p1, p2: bitflip probabilities for the first and second half of the bits
#         n_data: number of samples to generate
    
#     Returns:
#         (n_data, n) array of bitstrings
#     """
#     assert n % 2 == 0
#     bitstrings = np.random.rand(n_data, n) < np.concatenate([p1*np.ones(n//2), p2*np.ones(n//2)])
#     return bitstrings


def optimal_decoding(n, p1, p2):
    """brute-force compute the optimal decoding success probability for this biased bitflip model."""
    X, Y = create_dataset(n)
    probs = noise_model(Y, n, p1, p2)
    lookup = {}
    for x, w in zip(X, probs):
        # finding the maximum conditional p(y|x); for a classical code there are only 2 possible syndromes per error
        if tuple(x) in lookup:
            lookup[tuple(x)] = max(lookup[tuple(x)], w)
        else:
            lookup[tuple(x)] = w
    return sum(lookup.values())



def sample_histogram(probs, m):
    """Sample m bitstrings from a distribution defined by probs.

    This is used for "virtual training": since we have discrete data,
    instead of sampling `n_train` bitstrings for training data, we can
    just reweight a loss function according to a histogram of n_train 
    samples from probs.

    Args:
        probs: (2**n,) array of probabilities of each bitstring
        m: number of samples to draw
    """
    sample = np.random.choice(len(probs), m, p=probs)
    hist = np.zeros(len(probs))
    for s in sample:
        hist[s] += 1
    return hist / m

def sample_virtual_XY(probs, m, n, dataset_config):
    """Sample virtual dataset.
    
    The virtual dataset is a tuple (X, Y, weights).

    args:
        probs: (2**n,) array of probabilities of each bitstring
        m: number of samples to draw
        n: number of bits
        H: parity check matrix
    Returns:
        Y is (N, n) array, where N is the number of unique samples and n is the number of bits.
            Each row of Y is a unique bitstring that appeared in sampling.
        X is (N, n-1) array computed via the parity check matrix.
        weights: (N,) array of weights for each sample.
        hist: (2**n,) array of counts of each bitstring in the dataset, ordered correctly. This is 
            so that we can build a complete trainings et iteratively when batching training.

    """
    H = dataset_config['pcm']

    hist = sample_histogram(probs, m) 
    base = binarr(n)
    Y = np.array([base[i] for i in range(len(probs)) if hist[i] > 0])
    X = Y @ H.T % 2
    weights = np.array([hist[i] for i in range(len(probs)) if hist[i] > 0])
    weights = weights / weights.sum()
    Xb_tensor = torch.tensor(X, dtype=torch.float32)
    Yb_tensor = torch.tensor(Y, dtype=torch.float32)
    weightsb = torch.tensor(weights, dtype=torch.float32)

    return Xb_tensor, Yb_tensor, weightsb, hist


def make_good_examples(n, p1, p2):
    """prepare a dataset of specifically the good examples."""
    X_train, Y_train = create_dataset(n)
    probs = noise_model(Y_train, n, p1, p2)
    good_examples = {}
    for x, y, w in zip(X_train, Y_train, probs):
        if tuple(x) in good_examples:
            if w > good_examples[tuple(x)][1]:
                good_examples[tuple(x)] = (y, w)
        else:
            good_examples[tuple(x)] = (y, w)

    Y_train = np.array([v[0] for v in good_examples.values()])
    X_train = np.array([k for k in good_examples.keys()])
    probs = np.array([v[1] for v in good_examples.values()])
    probs = probs / probs.sum()
    return X_train, Y_train, probs