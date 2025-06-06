"""toy_problem_data.py - This module constructs the toy problem dataset"""

import numpy as np
import torch
from mldec.utils.bit_tools import binarr
from mldec.datasets import tools


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


def create_dataset_training(n, dataset_config):
    """
    sos_eos: Tuple (sos, eos) to append to the targets. If None, no tokens are appended.
    """
    X, Y = create_dataset(n)
    weights = noise_model(Y, n, dataset_config)
    sos_eos = dataset_config.get("sos_eos")
    if sos_eos:
        sos, eos = sos_eos
        Y = torch.cat([sos*torch.ones((Y.shape[0], 1)), Y, eos*torch.ones((Y.shape[0], 1))], axis=1)
    return X, Y, weights

        

def noise_model(s, n, dataset_config, permute=None, numpy=False):
    """Create the probability of array `s` under the biased bitflip model.
    
    The first n//2 bits have prob. p1 of flipping, the last n//2 have prob. p2.
    
    Warning: if the difference in bias is too much, the weight-ordering of bitstrings 
    is no longer the same as likelihood ordering. Manually check that there is no
    weight-(k+1) bitstring with higher likelihood than a weight-k bitstring!

    Args:
        s: (n_data, n) array of bitstrings.
        p, alpha: p1 = p is the bitflip probability for the first half, and p2 = alpha * p1 for the second half.

    """
    p, alpha = dataset_config['p'], dataset_config['alpha']
    p1 = p
    p2 = p * alpha
    if permute is not None:
        s = s[:,permute]
    if numpy:
        p_first = np.prod(p1*s[:,:n//2] +(1-p1)*(1-s[:,:n//2]), axis=1)
        p_second = np.prod(p2*s[:,n//2:] + (1-p2)*(1-s[:,n//2:]), axis=1)
        return np.multiply(p_first, p_second)
    
    p_first = torch.prod(p1*s[:,:n//2] +(1-p1)*(1-s[:,:n//2]), axis=1)
    p_second = torch.prod(p2*s[:,n//2:] + (1-p2)*(1-s[:,n//2:]), axis=1)
    return torch.multiply(p_first, p_second)


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


def sample_virtual_XY(probs, m, n, dataset_config):
    """Sample virtual dataset.
    
    The virtual dataset is a tuple (X, Y, weights). If m==1994, we simulate infinite training data.

    args:
        probs: (2**n,) array of probabilities of each bitstring
        m: number of samples to draw
        n: number of bits
        config: Should contain p1, p2, and the parity check matrix.
        sos_eos: Tuple (sos, eos) to append to the targets. If None, no tokens are appended.
    Returns:
        Y is (N, n) array, where N is the number of unique samples and n is the number of bits.
            Each row of Y is a unique bitstring that appeared in sampling.
        X is (N, n-1) array computed via the parity check matrix.
        weights: (N,) array of weights for each unique sample in the batch.
        hist: (2**n,) array of counts of each bitstring in the dataset, ordered correctly to match
            the ordering lf `probs`. This is used to build a complete training set iteratively during batching.

    """
    H = dataset_config['pcm']
    hist = tools.sample_histogram(probs, m) 
    base = binarr(n)

    if m == 1994:
        Y = base
        X = Y @ H.T % 2
        Xb_tensor = torch.tensor(X, dtype=torch.float32)
        Yb_tensor = torch.tensor(Y, dtype=torch.float32)
        sos_eos = dataset_config.get("sos_eos")
        if sos_eos:
            sos, eos = sos_eos
            Yb_tensor = torch.cat([sos*torch.ones((Yb_tensor.shape[0], 1)), Yb_tensor, eos*torch.ones((Yb_tensor.shape[0], 1))], axis=1)
        return Xb_tensor, Yb_tensor, torch.tensor(probs, dtype=torch.float32), torch.tensor(probs, dtype=torch.float32).numpy()

    Y = np.array([base[i] for i in range(len(probs)) if hist[i] > 0])
    X = Y @ H.T % 2
    weights = np.array([hist[i] for i in range(len(probs)) if hist[i] > 0])
    weights = weights / weights.sum()
    Xb_tensor = torch.tensor(X, dtype=torch.float32)
    Yb_tensor = torch.tensor(Y, dtype=torch.float32)
    sos_eos = dataset_config.get("sos_eos")
    if sos_eos:
        sos, eos = sos_eos
        Yb_tensor = torch.cat([sos*torch.ones((Yb_tensor.shape[0], 1)), Yb_tensor, eos*torch.ones((Yb_tensor.shape[0], 1))], axis=1)
    weightsb = torch.tensor(weights, dtype=torch.float32)

    return Xb_tensor, Yb_tensor, weightsb, hist


def uniform_over_good_examples(n, dataset_config):
    """prepare a dataset of specifically the good examples.
    
    Ties are broken arbitrarily.
    """
    X_train, Y_train = create_dataset(n)
    probs = noise_model(Y_train, n, dataset_config)
    good_examples = {}
    as_tuple = lambda x: tuple([int(i) for i in x]) # torch tensors should not be hashed.
    for j, (x, y, w) in enumerate(zip(X_train, Y_train, probs)):
        if as_tuple(x) in good_examples:
            if w > good_examples[as_tuple(x)][1]:
                good_examples[as_tuple(x)] = (j, w)

        else:
            good_examples[as_tuple(x)] = (j, w)
    indices_to_keep = [v[0] for v in good_examples.values()]
    probs = np.zeros_like(probs)
    probs[indices_to_keep] = 1
    probs = probs / probs.sum()
    probs = torch.tensor(probs, dtype=torch.float32)
    sos_eos = dataset_config.get("sos_eos")
    if sos_eos:
        sos, eos = sos_eos
        Y_train = torch.cat([sos*torch.ones((Y_train.shape[0], 1)), Y_train, eos*torch.ones((Y_train.shape[0], 1))], axis=1)
    return X_train, Y_train, probs